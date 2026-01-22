==============
Data Splitting
==============

This guide explains how Tiny ML Tensorlab splits data into training, validation,
and test sets.

Split Methods
-------------

There are two methods for splitting data:

**1. amongst_files (Default)**

Files are divided into different sets:

* File A goes to training
* File B goes to validation
* File C goes to testing

Good when each file represents a distinct experiment or session.

**2. within_files**

Each file is split internally:

* First 60% of File A → Training
* Next 30% of File A → Validation
* Last 10% of File A → Testing

Good when files contain long continuous sequences.

Configuration
-------------

**Using split_factor (Auto-split)**

.. code-block:: yaml

   dataset:
     dataset_name: 'my_data'
     input_data_path: '/path/to/data'
     split_type: 'amongst_files'     # or 'within_files'
     split_factor: [0.6, 0.3, 0.1]   # train, val, test ratios

Ratios must sum to 1.0.

**Using Annotation Files (Manual Split)**

Create files in ``annotations/`` folder:

.. code-block:: text

   my_dataset/
   ├── classes/ (or files/)
   │   └── ...
   └── annotations/
       ├── instances_train_list.txt
       ├── instances_val_list.txt
       └── instances_test_list.txt    # Optional

If annotations exist, they override ``split_factor``.

Annotation File Format
----------------------

List file paths relative to the data directory, one per line:

**instances_train_list.txt:**

.. code-block:: text

   class_A/sample1.csv
   class_A/sample2.csv
   class_B/sample1.csv
   class_B/sample2.csv

**instances_val_list.txt:**

.. code-block:: text

   class_A/sample3.csv
   class_B/sample3.csv

Split Examples
--------------

**Example 1: 10 Files, amongst_files, [0.6, 0.3, 0.1]**

.. code-block:: text

   Files: file1.csv through file10.csv

   Training (60%):   file1-6.csv (6 files)
   Validation (30%): file7-9.csv (3 files)
   Testing (10%):    file10.csv (1 file)

Each file retains all its rows.

**Example 2: 10 Files, within_files, [0.6, 0.3, 0.1]**

.. code-block:: text

   Each file has 100 rows

   Training:   Rows 0-59 from all 10 files
   Validation: Rows 60-89 from all 10 files
   Testing:    Rows 90-99 from all 10 files

All files appear in all sets, but different portions.

When to Use Each Method
-----------------------

**Use amongst_files when:**

* Each file is a separate experiment
* Files have different conditions (different subjects, machines)
* You want to test generalization to new experiments

**Use within_files when:**

* Files are very long continuous recordings
* You want maximum data utilization
* The data is homogeneous throughout

Best Practices
--------------

**1. Keep Test Data Truly Held-Out**

Never use test data during model development. Only evaluate on test set
when you've finalized your model.

**2. Use Stratified Splits for Classification**

For classification, try to maintain class proportions in each split.
The auto-split attempts this automatically.

**3. Consider Temporal Ordering**

For time series, consider whether random splitting makes sense.
Sometimes you want earlier data for training, later data for testing.

**4. Use Annotation Files for Reproducibility**

Manual annotation files ensure the same splits across runs.

Creating Annotation Files
-------------------------

**Automatically Generated**

Run training once without annotations, then find generated files in the
output directory.

**Manually Created**

Use a script to create deterministic splits:

.. code-block:: python

   import os
   import random

   def create_splits(data_dir, train_ratio=0.6, val_ratio=0.3):
       # List all files
       files = []
       for class_name in os.listdir(os.path.join(data_dir, 'classes')):
           class_dir = os.path.join(data_dir, 'classes', class_name)
           for f in os.listdir(class_dir):
               files.append(f'{class_name}/{f}')

       # Shuffle deterministically
       random.seed(42)
       random.shuffle(files)

       # Split
       n_train = int(len(files) * train_ratio)
       n_val = int(len(files) * val_ratio)

       train_files = files[:n_train]
       val_files = files[n_train:n_train + n_val]
       test_files = files[n_train + n_val:]

       # Write annotation files
       ann_dir = os.path.join(data_dir, 'annotations')
       os.makedirs(ann_dir, exist_ok=True)

       with open(os.path.join(ann_dir, 'instances_train_list.txt'), 'w') as f:
           f.write('\n'.join(train_files))

       with open(os.path.join(ann_dir, 'instances_val_list.txt'), 'w') as f:
           f.write('\n'.join(val_files))

       with open(os.path.join(ann_dir, 'instances_test_list.txt'), 'w') as f:
           f.write('\n'.join(test_files))

Verifying Splits
----------------

After training, check the log for split information:

.. code-block:: text

   Dataset loaded:
   - Training samples: 1200
   - Validation samples: 400
   - Test samples: 200
