=======================
Bring Your Own Data
=======================

This section explains how to format your datasets for use with Tiny ML ModelZoo.
The toolchain supports various data formats and automatically handles preprocessing.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   classification_format
   regression_format
   forecasting_format
   anomaly_detection_format
   audio_radar_format
   data_splitting

Dataset Format Overview
-----------------------

Tiny ML ModelZoo uses different folder structures depending on the task type:

**Classification Tasks**

.. code-block:: text

   dataset_name/
   ├── classes/
   │   ├── class1/
   │   │   ├── file1.csv
   │   │   └── file2.csv
   │   ├── class2/
   │   │   └── file1.csv
   │   └── classN/
   └── annotations/          # Optional - auto-generated if missing
       ├── instances_train_list.txt
       └── instances_val_list.txt

**Anomaly Detection Tasks**

.. code-block:: text

   dataset_name/
   ├── classes/
   │   ├── Normal/           # Training data (normal operation only)
   │   │   ├── file1.csv
   │   │   └── file2.csv
   │   └── Anomaly/          # Test-only data (fault/anomaly samples)
   │       ├── file1.csv
   │       └── file2.csv
   └── annotations/          # Optional - auto-generated if missing

See :doc:`anomaly_detection_format` for full details.

**Regression & Forecasting Tasks**

.. code-block:: text

   dataset_name/
   ├── files/               # MUST be named "files"
   │   ├── datafile1.csv
   │   └── datafileN.csv
   └── annotations/         # Required for these tasks
       ├── instances_train_list.txt
       └── instances_val_list.txt

**Audio and Radar Classification Tasks (new in 1.5.0)**

Both use the same ``classes/<class_name>/`` layout as time series classification.
Audio classes hold ``.wav``/``.flac`` recordings and radar classes hold ``.csv``/``.npy``
files. See :doc:`audio_radar_format`.

Splits are described in :doc:`data_splitting`. If no annotation files are found, the
split lists are generated automatically (``train``, ``val`` and ``test``).

Supported File Formats
----------------------

* **CSV files** (``.csv``) - Most common, human-readable
* **Text files** (``.txt``) - Same format as CSV
* **NumPy arrays** (``.npy``) - Binary format, faster loading
* **Pickle files** (``.pkl``) - Python serialized pandas DataFrames

The list above applies to time series tasks. Audio tasks read ``.wav`` and ``.flac``; radar tasks read
``.csv`` and ``.npy``.

Data Sources
------------

You can provide your dataset as:

* A **local directory** path
* A **local ZIP file** path (extracted next to the ZIP file)
* A **remote URL** to a ZIP file (automatically downloaded)

Example:

.. code-block:: yaml

   dataset:
     dataset_name: my_dataset
     input_data_path: '/path/to/dataset'  # or URL

For working configs see :doc:`/CONFIG_YAML_GUIDE`. Examples of every dataset layout are in the
:doc:`/examples/index` section.
