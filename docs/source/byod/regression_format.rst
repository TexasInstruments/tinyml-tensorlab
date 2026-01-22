===========================
Regression Dataset Format
===========================

This guide explains how to format datasets for time series regression tasks.

Directory Structure
-------------------

Regression datasets use a flat ``files/`` folder structure (not class folders):

.. code-block:: text

   my_dataset/
   ├── files/                          # MUST be named "files"
   │   ├── datafile1.csv
   │   ├── datafile2.csv
   │   └── datafileN.csv
   └── annotations/                    # Required for regression
       ├── instances_train_list.txt
       └── instances_val_list.txt

.. important::
   The data directory **must** be named ``files/``, not ``classes/`` or anything else.

Data File Format
----------------

**Critical**: The target value must be in the **last column**.

.. code-block:: text

   feature1,feature2,feature3,...,target
   0.5,18.2,45.5,...,0.187
   0.6,18.5,45.6,...,0.245
   ...

**Example: Motor Torque Prediction**

.. code-block:: text

   current_d,current_q,voltage_d,voltage_q,motor_speed,pm_temp,target_torque
   -0.450,0.032,18.805,1.499,0.002,24.55,0.187
   -0.325,0.045,18.818,1.542,0.003,24.54,0.245
   -0.440,0.028,18.876,1.456,0.002,24.54,0.176

* Columns 1-6: Input features
* Column 7 (last): Target value to predict

Time Column Handling
--------------------

Any column containing "time" is automatically dropped:

.. code-block:: text

   Time,feature1,feature2,target
   0.001,0.5,18.2,0.187
   0.002,0.6,18.5,0.245

The "Time" column will be removed automatically.

Annotation Files (Required)
---------------------------

Unlike classification, regression **requires** annotation files:

**instances_train_list.txt**:

.. code-block:: text

   datafile1.csv
   datafile3.csv
   datafile5.csv

**instances_val_list.txt**:

.. code-block:: text

   datafile2.csv
   datafile4.csv

**instances_test_list.txt** (optional):

.. code-block:: text

   datafile6.csv

If you don't provide annotations, ModelMaker will auto-generate them.

Configuration
-------------

.. code-block:: yaml

   dataset:
     enable: True
     dataset_name: 'my_regression_data'
     input_data_path: '/path/to/my_dataset'
     data_dir: 'files'
     annotation_dir: 'annotations'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']  # Required!
     frame_size: 128
     stride_size: 0.25
     variables: 6                            # Input columns (excluding target)

.. important::
   ``SimpleWindow`` transform is **required** for regression tasks.

Target Processing
-----------------

The target value (last column) is processed as follows:

1. Each window of ``frame_size`` rows is extracted
2. The target value is **averaged** across the window
3. This averaged value becomes the label for that window

Example with ``frame_size=4``:

.. code-block:: text

   Window 1: rows 0-3, targets [0.18, 0.24, 0.17, 0.19] → avg = 0.195
   Window 2: rows 2-5, targets [0.17, 0.19, 0.22, 0.20] → avg = 0.195

Complete Example
----------------

**Dataset structure:**

.. code-block:: text

   torque_measurement/
   ├── files/
   │   ├── experiment_001.csv
   │   ├── experiment_002.csv
   │   ├── experiment_003.csv
   │   └── experiment_004.csv
   └── annotations/
       ├── instances_train_list.txt    # experiment_001.csv, experiment_002.csv
       └── instances_val_list.txt      # experiment_003.csv, experiment_004.csv

**experiment_001.csv:**

.. code-block:: text

   i_d,i_q,u_d,u_q,speed,temp,torque
   -0.45,0.03,18.80,1.49,0.002,24.5,0.187
   -0.32,0.04,18.81,1.54,0.003,24.5,0.245
   ...

**config.yaml:**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_regression'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'torque_measurement'
     input_data_path: '/data/torque_measurement'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 128
     stride_size: 0.25
     variables: 6

   training:
     model_name: 'REGR_1k_NPU'
     training_epochs: 100

Common Issues
-------------

**"Target not found" error**

Ensure the target is in the last column of your CSV.

**"No windows generated" error**

Check that files have at least ``frame_size`` rows.

**Poor regression performance**

* Try different ``frame_size`` values
* Ensure input features are relevant to target
* Normalize extreme values in your data
