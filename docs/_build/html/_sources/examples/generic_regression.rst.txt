====================================
Generic Time Series Regression
====================================

This example serves as a **"Hello World" introduction** to time series regression
using the TinyML ModelMaker toolchain. It demonstrates how to use any generic
time series regression task with our toolchain.

Overview
--------

* **Task**: Time series regression (continuous value prediction)
* **Dataset**: Synthetic data (y = 1.2 sin(x) + 3.2 cos(x))
* **Model**: REGR_1k (~1,000 parameters)
* **Target**: F28P55 (NPU device)

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/generic_timseries_regression/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\generic_timseries_regression\config.yaml

Understanding the Dataset
-------------------------

The dataset consists of synthetically generated data:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``x``
     - Randomly generated in range [0, 3]
   * - ``y``
     - Target: y = 1.2 sin(x) + 3.2 cos(x)

**Dataset split:**

* Training: 7 files
* Validation: 2 files
* Test: 1 file

Each file contains 5000 datapoints.

Dataset Format
--------------

For regression tasks, ModelMaker expects the dataset in this structure:

.. code-block:: text

   {dataset_name}.zip/
   |
   |-- files/
   |     |-- {file1}.csv
   |     |-- {file2}.csv
   |     |-- {fileN}.csv
   |
   |-- annotations/
         |-- file_list.txt
         |-- instances_train_list.txt
         |-- instances_val_list.txt
         |-- instances_test_list.txt

.. note::

   Unlike classification tasks, regression **always requires annotation files**
   to specify train/validation/test splits.

Configuration
-------------

**common section:**

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F28P55

**dataset section:**

.. code-block:: yaml

   dataset:
     dataset_name: generic_timeseries_regression
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_regression.zip

**data_processing_feature_extraction section:**

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
       - SimpleWindow
     frame_size: 10
     stride_size: 1
     variables: 1

**training section:**

.. code-block:: yaml

   training:
     optimizer: adam
     lr_scheduler: cosineannealinglr
     model_name: REGR_1k
     batch_size: 128
     training_epochs: 100
     lambda_reg: 0.01
     num_gpus: 1
     quantization: 0

Evaluation Metrics
------------------

**RMSE (Root Mean Square Error)**

Measures the mean of the square root of the sum of squares of errors.

* Range: 0 to infinity
* Ideal value: 0 (lower is better)

**R2 Score (Coefficient of Determination)**

Indicates how well predictions match actual values.

* Range: (-infinity, 1]
* Ideal value: 1 (higher is better)

Expected Results
----------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Metric
     - Value
   * - RMSE
     - 0.13
   * - R2 Score
     - 0.99

Output Location
---------------

Results are saved to::

   ../tinyml-modelmaker/data/projects/{dataset_name}/run/{date-time}/{model_name}/

Key files:

* ``training/base/best_model.pt`` - Trained model
* ``training/base/post_training_analysis/`` - Prediction plots
* ``compilation/artifacts/mod.a`` - Compiled for device

Next Steps
----------

After successfully running this generic example:

1. Try :doc:`torque_measurement_regression` - Real-world motor torque prediction
2. Explore :doc:`/byod/regression_format` - Use your own regression data
3. Read :doc:`/getting_started/understanding_config` - Learn all config options
