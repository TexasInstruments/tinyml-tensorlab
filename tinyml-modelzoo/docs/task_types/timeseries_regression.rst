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
     model_name: 'REGR_2k_NPU'
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
   └── annotations/            # optional, generated if missing
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

Task Settings
-------------

.. list-table::
   :widths: 30 70

   * - ``task_type``
     - ``generic_timeseries_regression``
   * - ``target_module``
     - ``timeseries`` (inferred from ``task_type``)
   * - Data folder
     - ``files/``
   * - Early stopping
     - On by default in 1.5.0; see :doc:`/features/early_stopping`

Available Models
----------------

Defined in ``tinyml_modelzoo/model_descriptions/regression.py``. The number in the name is the
approximate parameter count.

* **NPU models**: ``REGR_500_NPU``, ``REGR_2k_NPU``, ``REGR_6k_NPU``, ``REGR_8k_NPU``, ``REGR_20k_NPU``
* **Standard models**: ``REGR_1k``, ``REGR_2k``, ``REGR_3k``, ``REGR_4k``, ``REGR_10k``, ``REGR_13k``

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

   (illustrative, not from a real run)
   Float32 Model:
   R² Score: <value>
   MSE: <value>

   Quantized Model:
   R² Score: <value>
   MSE: <value>

The log reports validation MSE and R2-Score for ``FloatTrain`` and ``QuantTrain``, the same for the best
epoch, and RMSE and R2-Score on the test data. See :doc:`/getting_started/understanding_run_output`.
Example: Torque Measurement
---------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_regression'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'torque_measurement'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/torque_measurement_dataset.zip'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 128
     stride_size: 0.25
     variables: 10

   training:
     model_name: 'REGR_2k_NPU'
     training_epochs: 50

This follows ``examples/torque_measurement_regression/config.yaml``, which uses ``REGR_10k``
(batch size 512, learning rate 0.001, early stopping on). Run it with
``./run_tinyml_modelzoo.sh examples/torque_measurement_regression/config.yaml``.

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
* :doc:`/examples/generic_regression` - Generic regression example
* :doc:`/examples/torque_measurement_regression` - Torque measurement example
* :doc:`/examples/washing_machine_regression` - Washing machine load weighing
