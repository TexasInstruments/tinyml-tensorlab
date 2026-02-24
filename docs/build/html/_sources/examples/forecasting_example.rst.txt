===================
Forecasting Example
===================

This example demonstrates time series forecasting to predict future
values based on historical patterns.

Overview
--------

* **Task**: Time series forecasting
* **Application**: Predict next N values in a sequence
* **Model**: Regression-based forecasting
* **Use cases**: Predictive control, resource planning

When to Use Forecasting
-----------------------

Forecasting is useful when you need to:

* Predict future sensor values
* Anticipate system behavior
* Enable proactive control decisions
* Implement look-ahead algorithms

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/forecasting/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\\forecasting\\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_forecasting'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'forecasting_example_dsg'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_RAW_256Feature_1Frame'
     variables: 1
     target_column: 0    # Which column to forecast
     forecast_horizon: 10  # Predict 10 steps ahead

   training:
     model_name: 'FCST_4k_NPU'
     training_epochs: 50
     batch_size: 128

   testing:
     enable: True

   compilation:
     enable: True

How Forecasting Works
---------------------

**Training:**

The model learns patterns from historical data:

.. code-block:: text

   Input window: [x(t-N), x(t-N+1), ..., x(t-1), x(t)]
   Target:       [x(t+1), x(t+2), ..., x(t+H)]

   where N = input window size, H = forecast horizon

**Inference:**

Given recent history, predict future values:

.. code-block:: text

   Recent data: [10.2, 10.5, 10.8, 11.0, 11.3]
   Prediction:  [11.5, 11.7, 11.9]  (next 3 values)

Dataset Format
--------------

Forecasting uses the same format as regression:

.. code-block:: text

   my_forecasting_dataset/
   ├── annotations.yaml
   └── files/
       ├── sequence_001.csv
       ├── sequence_002.csv
       └── ...

Each CSV contains continuous time series:

.. code-block:: text

   timestamp,temperature,pressure,flow
   0.000,25.1,101.3,50.2
   0.001,25.2,101.2,50.1
   0.002,25.3,101.4,50.3
   ...

Configuration for dataset:

.. code-block:: yaml

   data_processing_feature_extraction:
     target_column: 0      # Forecast 'temperature' (column 0)
     forecast_horizon: 10  # Predict 10 steps ahead
     variables: 3          # Use all 3 variables as input

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Parameters
     - Description
   * - ``FCST_500_NPU``
     - ~500
     - Simple patterns
   * - ``FCST_1k_NPU``
     - ~1,000
     - Light forecasting
   * - ``FCST_2k_NPU``
     - ~2,000
     - Balanced
   * - ``FCST_4k_NPU``
     - ~4,000
     - Complex patterns
   * - ``FCST_8k_NPU``
     - ~8,000
     - High complexity

**Model Selection:**

* Short-term, simple patterns: ``FCST_500_NPU``
* General purpose: ``FCST_2k_NPU``
* Long-term or complex: ``FCST_4k_NPU`` or larger

Expected Results
----------------

.. code-block:: text

   Training complete.

   Test Set Results:
   MSE: 0.023
   MAE: 0.12
   R²: 0.95

   Forecast Horizon Performance:
   Step 1: MAE=0.08
   Step 5: MAE=0.15
   Step 10: MAE=0.22

Key Metrics
-----------

**Mean Squared Error (MSE):**

Average squared difference between predicted and actual:

* Lower is better
* Sensitive to outliers

**Mean Absolute Error (MAE):**

Average absolute difference:

* Lower is better
* More interpretable than MSE

**R² Score:**

Proportion of variance explained:

* 1.0 = perfect prediction
* 0.0 = predicting mean
* Can be negative for bad models

Forecast Horizon Trade-offs
---------------------------

Longer forecast horizons are harder:

.. code-block:: yaml

   # Easy: predict 1 step ahead
   data_processing_feature_extraction:
     forecast_horizon: 1

   # Moderate: predict 10 steps ahead
   data_processing_feature_extraction:
     forecast_horizon: 10

   # Hard: predict 50 steps ahead
   data_processing_feature_extraction:
     forecast_horizon: 50

**Tips for longer horizons:**

* Use larger models
* Increase input window size
* Include more relevant features
* Accept higher error for distant predictions

Multi-Variable Forecasting
--------------------------

Use multiple input variables to improve predictions:

.. code-block:: yaml

   data_processing_feature_extraction:
     variables: 3          # temp, pressure, flow
     target_column: 0      # Forecast temperature
     forecast_horizon: 10

The model uses all variables as inputs but predicts only the target.

Feature Extraction Options
--------------------------

**Raw Time Domain:**

Best for smooth, continuous signals:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_RAW_256Feature_1Frame'

**FFT Frequency Domain:**

Best for periodic signals:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_FFTBIN_64Feature_4Frame'

**Multi-Frame:**

Captures longer temporal context:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_128Input_RAW_128Feature_4Frame'

Practical Applications
----------------------

**Temperature Prediction:**

Predict future temperature for proactive cooling:

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_forecasting'
     target_device: 'F28P55'

   data_processing_feature_extraction:
     variables: 1
     target_column: 0
     forecast_horizon: 20  # 20 samples ahead

   training:
     model_name: 'FCST_2k_NPU'

**Load Forecasting:**

Predict power demand for grid management:

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_forecasting'

   data_processing_feature_extraction:
     variables: 4  # load, temperature, time, day
     target_column: 0  # Forecast load
     forecast_horizon: 60  # 1 hour ahead (1-min samples)

   training:
     model_name: 'FCST_4k_NPU'

**Motion Prediction:**

Predict trajectory for control systems:

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_forecasting'

   data_processing_feature_extraction:
     variables: 6  # position x,y,z and velocity x,y,z
     target_column: 0  # Forecast x position
     forecast_horizon: 5

   training:
     model_name: 'FCST_2k_NPU'

Deployment Considerations
-------------------------

**Inference Frequency:**

* Run forecasting at regular intervals
* Update predictions as new data arrives
* Use sliding window approach

**Confidence Estimation:**

* Training MSE provides baseline error estimate
* Actual error may vary with input
* Consider ensemble approaches for uncertainty

**Horizon-Dependent Actions:**

* Use near-term predictions for immediate control
* Use far-term predictions for planning
* Weight decisions by prediction confidence

Troubleshooting
---------------

**High prediction error:**

* Increase model size
* Add more relevant input features
* Reduce forecast horizon
* Check data quality and preprocessing

**Model predicts constant value:**

* Learning rate may be too low
* Training data may lack variation
* Try different feature extraction

**Oscillating predictions:**

* May be overfitting
* Increase regularization
* Reduce model complexity

Comparison with Regression
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - Forecasting
     - Regression
   * - Output
     - Future sequence values
     - Single value from window
   * - Use case
     - Predict what comes next
     - Map input to output
   * - Target
     - Same variable, future time
     - Any related output

Next Steps
----------

* Review :doc:`/task_types/timeseries_forecasting` for details
* Learn about :doc:`/byod/forecasting_format` for your data
* Explore :doc:`/features/feature_extraction` options
