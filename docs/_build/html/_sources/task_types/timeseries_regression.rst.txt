======================
Time Series Regression
======================

Time series regression predicts continuous numerical values from sequences of
time-ordered data.

Overview
--------

**What it does**: Takes a sequence of sensor readings and outputs a continuous value.

**Example**: Motor currents + temperature → Torque (Nm)

**Use cases**:

* Torque estimation
* Motor speed prediction
* Load measurement
* Temperature estimation
* Power consumption prediction

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_regression'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'my_regression_data'
     input_data_path: '/path/to/data'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 128
     stride_size: 0.25
     variables: 10

   training:
     model_name: 'REGR_1k_NPU'
     training_epochs: 100

   testing: {}
   compilation: {}

Dataset Format
--------------

Regression datasets use a ``files/`` folder structure:

.. code-block:: text

   my_dataset/
   ├── files/
   │   ├── data1.csv
   │   ├── data2.csv
   │   └── ...
   └── annotations/
       ├── instances_train_list.txt
       └── instances_val_list.txt

**Important**: The target value must be in the **last column** of each CSV file.

Example CSV:

.. code-block:: text

   current_d,current_q,voltage,temperature,target_torque
   0.5,-0.3,18.2,45.5,0.187
   0.6,-0.2,18.5,45.6,0.245
   ...

See :doc:`/byod/regression_format` for details.

Available Models
----------------

**NPU-Optimized Models**:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model Name
     - Parameters
     - Use Case
   * - ``REGR_500_NPU``
     - ~500
     - Small
   * - ``REGR_1k_NPU``
     - ~1,000
     - Medium
   * - ``REGR_4k_NPU``
     - ~4,000
     - Large
   * - ``REGR_13k_NPU``
     - ~13,000
     - Very large

**Standard Models**:

* ``REGR_1k`` to ``REGR_13k`` - Non-NPU variants

Key Configuration
-----------------

**SimpleWindow is Required**:

Regression always requires windowing to create fixed-size inputs:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 128      # Window size
     stride_size: 0.25    # 75% overlap

**Target Processing**:

The target value (last column) is averaged across the window frame.

Metrics
-------

Regression models are evaluated using:

* **MSE** - Mean Squared Error
* **R² Score** - Coefficient of determination (1.0 = perfect)

Example output:

.. code-block:: text

   Float32 Model:
   R² Score: 0.994
   MSE: 0.0012

   Quantized Model:
   R² Score: 0.963
   MSE: 0.0089

Example: Torque Measurement
---------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_regression'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'torque_measurement'
     input_data_path: '/path/to/torque_data'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 128
     stride_size: 0.25
     variables: 10

   training:
     model_name: 'REGR_1k_NPU'
     training_epochs: 100

Tips
----

* Normalize your input features for better convergence
* Use sufficient ``frame_size`` to capture relevant patterns
* Regression typically needs more epochs than classification
* R² > 0.95 indicates good model fit
* Watch for overfitting if train R² >> val R²

See Also
--------

* :doc:`/byod/regression_format` - Dataset format
* :doc:`timeseries_forecasting` - Related task type
