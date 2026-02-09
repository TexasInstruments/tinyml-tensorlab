=====================================
Generic Time Series Forecasting
=====================================

This example serves as a **"Hello World" introduction** to time series forecasting
using the TinyML ModelMaker toolchain. It demonstrates how to use any generic
time series forecasting task with our toolchain.

Overview
--------

* **Task**: Time series forecasting (predict future values)
* **Dataset**: Simulated thermostat data (temperature oscillation)
* **Model**: FCST_LSTM10 (~542 parameters)
* **Target**: F28P55 (NPU device)

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/generic_timeseries_forecasting/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\generic_timeseries_forecasting\config.yaml

Understanding the Dataset
-------------------------

The dataset models a room temperature controlled by an ON/OFF heater with hysteresis:

* **Heater turns ON** when temperature drops below 20C
* **Heater turns OFF** when temperature rises above 24C
* Temperature changes gradually due to thermal inertia

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``timestamp``
     - Time index
   * - ``temperature``
     - Room temperature (C)

**Dataset split:**

* Training: 10 files (thermostat_01.csv to thermostat_10.csv)
* Validation: 2 files
* Test: 3 files

Each file contains 1000 timesteps of simulated temperature data.

Dataset Format
--------------

For forecasting tasks, ModelMaker expects the dataset in this structure:

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

   Unlike classification tasks, forecasting **always requires annotation files**
   to specify train/validation/test splits.

Configuration
-------------

**common section:**

.. code-block:: yaml

   common:
     task_type: generic_timeseries_forecasting
     target_device: F28P55

**dataset section:**

.. code-block:: yaml

   dataset:
     dataset_name: generic_timeseries_forecasting
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_forecasting.zip

**data_processing_feature_extraction section:**

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
       - SimpleWindow  # Mandatory for forecasting
     frame_size: 32
     stride_size: 0.1
     forecast_horizon: 2  # Predict 2 future timesteps
     variables: 1
     target_variables:
       - 0  # Column index (temperature)

**training section:**

.. code-block:: yaml

   training:
     model_name: FCST_LSTM10
     batch_size: 32
     training_epochs: 50
     num_gpus: 1
     quantization: 1
     optimizer: adam
     output_int: false  # Must be false for forecasting

.. important::

   ``output_int`` must be set to ``false`` for forecasting tasks.

Target Variables
----------------

The ``target_variables`` parameter can be specified in multiple formats:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Format
     - Example
     - Description
   * - Empty list
     - ``[]``
     - Predict all columns
   * - List of indices
     - ``[0]`` or ``[0, 1, 2]``
     - Column indices (0-indexed)
   * - List of names
     - ``['temperature']``
     - Column names from header
   * - Single index
     - ``0``
     - Single column index
   * - Single name
     - ``temperature``
     - Single column name

Evaluation Metrics
------------------

**SMAPE (Symmetric Mean Absolute Percentage Error)**

Measures percentage error between predicted and actual values.

* Range: 0% to 200%
* Ideal value: 0% (lower is better)

**R2 Score (Coefficient of Determination)**

Indicates how well predictions match actual values.

* Range: (-infinity, 1]
* Ideal value: 1 (higher is better)

Expected Results
----------------

**Float Training (Best Epoch 48):**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Variable
     - Timestep 1
     - Timestep 2
     - Overall
   * - SMAPE
     - 0.60%
     - 0.88%
     - 0.74%
   * - R2
     - 0.9811
     - 0.9563
     - 0.9687

**Test Results:**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Variable
     - Timestep 1
     - Timestep 2
     - Overall
   * - SMAPE
     - 0.66%
     - 0.95%
     - 0.80%
   * - R2
     - 0.9763
     - 0.9517
     - 0.9640

Output Location
---------------

Results are saved to::

   ../tinyml-modelmaker/data/projects/{dataset_name}/run/{date-time}/{model_name}/

Key outputs:

* ``training/base/best_epoch_{N}_results/`` - Float training results
* ``training/quantization/best_epoch_{N}_results/`` - Quantized results
* ``training/quantization/test_results/`` - Test results with prediction plots
* ``compilation/artifacts/mod.a`` - Compiled for device

Next Steps
----------

After successfully running this generic example:

1. Try :doc:`forecasting_pmsm_rotor` - Real-world motor temperature forecasting
2. Try :doc:`hvac_indoor_temp_forecast` - HVAC temperature prediction
3. Explore :doc:`/byod/forecasting_format` - Use your own forecasting data
4. Read :doc:`/getting_started/understanding_config` - Learn all config options
