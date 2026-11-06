=======================
Time Series Forecasting
=======================

Time series forecasting predicts future values of a time series based on
historical patterns.

Overview
--------

**What it does**: Takes historical values and predicts future values.

**Example**: Past 3 temperature readings → Next temperature reading

**Use cases**:

* Temperature prediction (motor, HVAC)
* Demand forecasting
* Energy consumption prediction
* Equipment degradation prediction

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_forecasting'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'my_forecast_data'
     input_data_path: '/path/to/data'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 3               # Lookback window
     forecast_horizon: 1         # Steps to predict ahead
     variables: [0, 5]           # Input columns
     target_variables: [5]       # Column to forecast
     stride_size: 0.4

   training:
     model_name: 'FCST_LSTM8'
     training_epochs: 100        # early stopping may end training sooner
     output_int: False           # Required for forecasting

   testing: {}
   compilation: {}

Key Parameters
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``frame_size``
     - Lookback window size (how many past values to use)
   * - ``forecast_horizon``
     - How many steps ahead to predict
   * - ``variables``
     - Which columns to use as inputs
   * - ``target_variables``
     - Which column(s) to forecast
   * - ``output_int``
     - **Must be False** for forecasting

**Variable Specification:**

.. code-block:: yaml

   # By index (0-based, after dropping time column)
   variables: [0, 5]
   target_variables: [5]

   # Or by column name
   variables: ['ambient', 'pm_temp']
   target_variables: ['pm_temp']

Dataset Format
--------------

Same as regression - uses ``files/`` folder:

.. code-block:: text

   my_dataset/
   ├── files/
   │   ├── sequence1.csv
   │   ├── sequence2.csv
   │   └── ...
   └── annotations/            # optional, generated if missing
       ├── instances_train_list.txt
       └── instances_val_list.txt

Example CSV:

.. code-block:: text

   ambient,coolant,current,pm_temp
   19.85,18.81,2.28,22.93
   19.85,18.79,2.28,22.94
   19.85,18.79,2.28,22.94
   ...

See :doc:`/byod/forecasting_format` for details.

Task Settings
-------------

.. list-table::
   :widths: 30 70

   * - ``task_type``
     - ``generic_timeseries_forecasting``
   * - ``target_module``
     - ``timeseries`` (inferred from ``task_type``)
   * - Data folder
     - ``files/``
   * - Early stopping
     - On by default in 1.5.0; see :doc:`/features/early_stopping`

Available Models
----------------

Defined in ``tinyml_modelzoo/model_descriptions/forecasting.py``:

* **NPU models**: ``FCST_500_NPU``, ``FCST_1k_NPU``, ``FCST_2k_NPU``, ``FCST_4k_NPU``, ``FCST_6k_NPU``,
  ``FCST_8k_NPU``, ``FCST_10k_NPU``, ``FCST_20k_NPU``
* **Standard CNN models**: ``FCST_3k``, ``FCST_13k``
* **LSTM models** (not NPU-compatible): ``FCST_LSTM8``, ``FCST_LSTM10`` - better for complex temporal patterns

Windowing Example
-----------------

With ``frame_size=3`` and ``forecast_horizon=1``:

.. code-block:: text

   Input: [t0, t1, t2] → Output: [t3]
   Input: [t1, t2, t3] → Output: [t4]
   Input: [t2, t3, t4] → Output: [t5]

Metrics
-------

* **SMAPE** - Symmetric Mean Absolute Percentage Error
* **R² Score** - Coefficient of determination

Lower SMAPE and higher R² are better. Both are logged for ``FloatTrain`` and ``QuantTrain`` and for the test
data; see :doc:`/getting_started/understanding_run_output`.

Example: PMSM Temperature Forecasting
-------------------------------------

From ``examples/forecasting_pmsm_rotor_temp/config.yaml``:

.. code-block:: yaml

   common:
     task_type: generic_timeseries_forecasting
     target_device: F28P55

   dataset:
     dataset_name: forecasting_pmsm_rotor_temp
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/pmsm_rotor_temp.zip

   data_processing_feature_extraction:
     data_proc_transforms: [SimpleWindow]
     frame_size: 3
     stride_size: 0.4
     forecast_horizon: 1
     variables: 6            # the first 6 columns (after any time column) are inputs
     target_variables: [5]   # forecast column 5 (pm temperature)

   training:
     model_name: FCST_LSTM8
     training_epochs: 20
     output_int: false

``variables`` may be an integer (use the first N columns) or a list of indices or names.

Important Notes
---------------

.. warning::
   * ``output_int`` must be ``False`` for forecasting
   * Feature extraction (FFT, etc.) is **not supported** for forecasting
   * Use raw time series only

Tips
----

* Start with small ``frame_size`` and increase if needed
* LSTM models often work better for temporal patterns
* Ensure target variable is included in input variables
* Normalize your data for better convergence

See Also
--------

* :doc:`/byod/forecasting_format` - Dataset format
* :doc:`timeseries_regression` - Related task type
* :doc:`/examples/generic_forecasting` - Generic forecasting example
* :doc:`/examples/forecasting_pmsm_rotor` - PMSM rotor temperature
* :doc:`/examples/hvac_indoor_temp_forecast` - HVAC indoor temperature
