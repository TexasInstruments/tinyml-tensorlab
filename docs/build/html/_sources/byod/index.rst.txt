=======================
Bring Your Own Data
=======================

This section explains how to format your datasets for use with Tiny ML Tensorlab.
The toolchain supports various data formats and automatically handles preprocessing.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   classification_format
   regression_format
   forecasting_format
   data_splitting

Dataset Format Overview
-----------------------

Tiny ML Tensorlab uses different folder structures depending on the task type:

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

**Regression & Forecasting Tasks**

.. code-block:: text

   dataset_name/
   ├── files/               # MUST be named "files"
   │   ├── datafile1.csv
   │   └── datafileN.csv
   └── annotations/         # Required for these tasks
       ├── instances_train_list.txt
       └── instances_val_list.txt

Supported File Formats
----------------------

* **CSV files** (``.csv``) - Most common, human-readable
* **Text files** (``.txt``) - Same format as CSV
* **NumPy arrays** (``.npy``) - Binary format, faster loading
* **Pickle files** (``.pkl``) - Python serialized pandas DataFrames

Data Sources
------------

You can provide your dataset as:

* A **local directory** path
* A **local ZIP file** path
* A **remote URL** to a ZIP file (automatically downloaded)

Example:

.. code-block:: yaml

   dataset:
     dataset_name: my_dataset
     input_data_path: '/path/to/dataset'  # or URL
