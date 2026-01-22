============================
Forecasting Dataset Format
============================

This guide explains how to format datasets for time series forecasting tasks.

Directory Structure
-------------------

Forecasting uses the same structure as regression:

.. code-block:: text

   my_dataset/
   ├── files/                          # MUST be named "files"
   │   ├── sequence1.csv
   │   ├── sequence2.csv
   │   └── sequenceN.csv
   └── annotations/                    # Required
       ├── instances_train_list.txt
       └── instances_val_list.txt

Data File Format
----------------

All variables (features) are in columns. You specify which to use as inputs
and which to forecast via configuration.

**Example: Temperature Forecasting**

.. code-block:: text

   ambient,coolant,current,pm_temp
   19.85,18.81,2.28,22.93
   19.85,18.79,2.28,22.94
   19.85,18.79,2.28,22.94
   19.85,18.77,2.28,22.94
   ...

In this example:

* Column 0 (``ambient``): Input feature
* Column 1 (``coolant``): Input feature
* Column 2 (``current``): Input feature
* Column 3 (``pm_temp``): Target to forecast

Key Difference from Regression
------------------------------

**Regression**: Target is a separate value for each window (in last column)

**Forecasting**: Target is a future value of an existing variable

.. code-block:: text

   Regression: Input [t0...t3] → Predict separate_target_avg
   Forecasting: Input [t0, t1, t2] → Predict variable[t3]

Configuration
-------------

.. code-block:: yaml

   dataset:
     enable: True
     dataset_name: 'my_forecast_data'
     input_data_path: '/path/to/my_dataset'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 3                    # Lookback (use 3 past values)
     forecast_horizon: 1              # Predict 1 step ahead
     stride_size: 0.4

     # Specify columns by index or name
     variables: [0, 3]                # Use columns 0 and 3 as inputs
     target_variables: [3]            # Forecast column 3

   training:
     model_name: 'FCST_LSTM8'
     output_int: False                # Required for forecasting!

Variable Specification Options
------------------------------

**By Column Index** (0-based, after time column removal):

.. code-block:: yaml

   variables: [0, 3]           # Use columns 0 and 3
   target_variables: [3]       # Forecast column 3

**By Column Name** (requires CSV header):

.. code-block:: yaml

   variables: ['ambient', 'pm_temp']
   target_variables: ['pm_temp']

**Multiple Targets** (forecast several variables):

.. code-block:: yaml

   variables: [0, 1, 2, 3]
   target_variables: [2, 3]     # Forecast columns 2 and 3

Windowing Behavior
------------------

With ``frame_size=3`` and ``forecast_horizon=1``:

.. code-block:: text

   Data: [v0, v1, v2, v3, v4, v5, v6, ...]

   Window 1: Input [v0, v1, v2] → Output [v3]
   Window 2: Input [v1, v2, v3] → Output [v4]
   Window 3: Input [v2, v3, v4] → Output [v5]
   ...

Complete Example
----------------

**Dataset structure:**

.. code-block:: text

   pmsm_temp_forecast/
   ├── files/
   │   ├── profile_10.csv
   │   ├── profile_11.csv
   │   └── profile_12.csv
   └── annotations/
       ├── instances_train_list.txt
       └── instances_val_list.txt

**profile_10.csv:**

.. code-block:: text

   ambient,coolant,u_d,u_q,i_a,pm
   19.850,18.815,1.499,0.032,2.281,22.936
   19.850,18.793,1.542,-0.092,2.281,22.941
   19.850,18.790,1.456,0.081,2.281,22.944
   ...

**config.yaml:**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_forecasting'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'pmsm_temp'
     input_data_path: '/data/pmsm_temp_forecast'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 3
     forecast_horizon: 1
     stride_size: 0.4
     variables: ['ambient', 'pm']      # Use ambient and pm as inputs
     target_variables: ['pm']           # Forecast pm temperature

   training:
     model_name: 'FCST_LSTM8'
     output_int: False

Important Notes
---------------

.. warning::
   * ``output_int`` must be ``False`` for forecasting
   * Feature extraction (FFT, wavelets) is **not supported**
   * The target variable should typically be included in input variables

Minimum Data Requirements
-------------------------

Each file must have at least::

   frame_size + forecast_horizon

rows to generate at least one training sample.

Common Issues
-------------

**"Insufficient sequence length" error**

Files need at least ``frame_size + forecast_horizon`` rows.

**Poor forecasting performance**

* Increase ``frame_size`` to capture more history
* Try LSTM models for complex temporal patterns
* Ensure sufficient training data
