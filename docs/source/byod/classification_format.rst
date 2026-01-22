=============================
Classification Dataset Format
=============================

This guide explains how to format datasets for time series classification tasks.

Directory Structure
-------------------

Classification datasets use a ``classes/`` folder where each subfolder represents
a class:

.. code-block:: text

   my_dataset/
   └── classes/
       ├── class_A/
       │   ├── sample1.csv
       │   ├── sample2.csv
       │   └── sample3.csv
       ├── class_B/
       │   ├── sample1.csv
       │   └── sample2.csv
       └── class_C/
           └── sample1.csv

**Key points**:

* Folder names become class labels
* Each CSV file is one sample (or multiple samples if using windowing)
* All files should have the same number of columns

Data File Format
----------------

**Headerless Format (Simple)**

Just numeric values, one measurement per row:

.. code-block:: text

   0.523
   0.612
   0.498
   0.701
   ...

**Headered Format (Recommended for Multi-Variable)**

First row contains column names:

.. code-block:: text

   channel_x,channel_y,channel_z
   0.523,0.112,-0.234
   0.612,0.098,-0.198
   0.498,0.145,-0.267
   ...

**Time Column Handling**

Any column containing "time" (case-insensitive) is automatically dropped:

.. code-block:: text

   Time,value1,value2
   0.001,0.523,0.112
   0.002,0.612,0.098
   ...

The "Time" column will be removed, leaving only value1 and value2.

Supported File Types
--------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extension
     - Description
   * - ``.csv``
     - Comma-separated values (most common)
   * - ``.txt``
     - Tab or space-separated text
   * - ``.npy``
     - NumPy array (binary, faster loading)
   * - ``.pkl``
     - Pickled pandas DataFrame

Annotations (Optional)
----------------------

You can optionally provide train/val/test splits using annotation files:

.. code-block:: text

   my_dataset/
   ├── classes/
   │   └── ...
   └── annotations/
       ├── file_list.txt              # All files (auto-generated if missing)
       ├── instances_train_list.txt   # Training files
       ├── instances_val_list.txt     # Validation files
       └── instances_test_list.txt    # Test files (optional)

**File List Format**

Each annotation file lists relative paths, one per line:

.. code-block:: text

   # instances_train_list.txt
   class_A/sample1.csv
   class_A/sample2.csv
   class_B/sample1.csv

If annotations folder is missing, ModelMaker auto-generates splits using
``split_factor`` from config.

Configuration
-------------

.. code-block:: yaml

   dataset:
     enable: True
     dataset_name: 'my_classification_data'
     input_data_path: '/path/to/my_dataset'  # or URL to .zip
     data_dir: 'classes'              # Default
     annotation_dir: 'annotations'    # Default (optional)
     split_type: 'amongst_files'
     split_factor: [0.6, 0.3, 0.1]    # train, val, test

   data_processing_feature_extraction:
     variables: 3                     # Number of data columns

Example: 3-Class Vibration Data
-------------------------------

Dataset structure:

.. code-block:: text

   vibration_dataset/
   └── classes/
       ├── normal/
       │   ├── run1.csv
       │   ├── run2.csv
       │   └── run3.csv
       ├── fault_type_A/
       │   ├── fault1.csv
       │   └── fault2.csv
       └── fault_type_B/
           └── fault1.csv

Sample file (``normal/run1.csv``):

.. code-block:: text

   accel_x,accel_y,accel_z
   0.012,0.005,-0.982
   0.015,0.008,-0.979
   0.010,0.003,-0.985
   ...

Config:

.. code-block:: yaml

   dataset:
     dataset_name: 'vibration_data'
     input_data_path: '/data/vibration_dataset'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_FFTBIN_16Feature_8Frame'
     variables: 3    # x, y, z axes

Class Balancing
---------------

For best results, try to have similar sample counts per class.

If classes are imbalanced:

* Collect more data for minority classes
* Use data augmentation (gain variation)
* Adjust training parameters

.. code-block:: yaml

   data_processing_feature_extraction:
     gain_variations: {fault_type_A: [0.9, 1.1]}  # Augment minority class

Common Issues
-------------

**"Dimension mismatch" error**

All files must have the same number of columns. Check for:

* Extra header rows
* Missing columns in some files
* Different delimiters

**"Empty file" error**

Ensure files contain actual data, not just headers.

**Class not detected**

* Check folder names don't contain special characters
* Ensure files exist in class folders
* Verify file extensions are supported
