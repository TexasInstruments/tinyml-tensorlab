==========================
Forecasting Examples Guide
==========================

In 1.4.0 this page was a generic forecasting tutorial that pointed to an ``examples/forecasting`` folder. That
folder does not exist in 1.5.0. This page is now a short guide to the forecasting examples and the settings
that matter for them. Each example has its own page with the full configuration.

Forecasting Examples in 1.5.0
-----------------------------

All use ``task_type: generic_timeseries_forecasting`` and are run from the ``tinyml-modelzoo`` folder with
``./run_tinyml_modelzoo.sh examples/<folder>/<config>.yaml`` (Windows: see :doc:`/getting_started/running_examples`).

.. list-table::
   :header-rows: 1
   :widths: 40 14 18 28

   * - Config
     - Device
     - Model
     - Page
   * - ``generic_timeseries_forecasting/config.yaml``
     - F28P55
     - ``FCST_LSTM10``
     - :doc:`generic_forecasting`
   * - ``forecasting_pmsm_rotor_temp/config.yaml``
     - F28P55
     - ``FCST_LSTM8``
     - :doc:`forecasting_pmsm_rotor`
   * - ``forecasting_pmsm_rotor_temp/config_MSPM0.yaml``
     - MSPM0G5187
     - ``FCST_LSTM8``
     - :doc:`forecasting_pmsm_rotor`
   * - ``hvac_indoor_temp_forecast/config.yaml``
     - F28P55
     - ``FCST_LSTM10``
     - :doc:`hvac_indoor_temp_forecast`

How Forecasting Works
---------------------

A forecasting model reads a window of past values (``frame_size`` time steps of ``variables`` input columns)
and predicts the next ``forecast_horizon`` steps of one or more ``target_variables``. The ``SimpleWindow``
transform creates the windows and is mandatory. The stride (``stride_size``) is a fraction of the frame size.

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     frame_size: 32
     stride_size: 0.1
     forecast_horizon: 2
     variables: 1
     target_variables:
     - 0

Rules to remember:

* The dataset needs the ``files/`` plus ``annotations/`` layout, see :doc:`/byod/forecasting_format`.
* ``output_int`` must be ``false`` in the ``training`` section.
* Feature extraction is not supported for forecasting. ``Downsample`` can be added to ``data_proc_transforms``
  together with ``sampling_rate`` and ``new_sr``.
* ``target_variables`` accepts column indices (0-based after the time column), column names, a single index or
  name, or an empty list to predict all columns.
* Multi-variable forecasting works by listing several indices or names in ``target_variables``. Metrics are
  reported per variable and per predicted time step.

Metrics
-------

SMAPE (0% to 200%, lower is better) and R2 (higher is better) are computed per target variable and per predicted
time step. The best epoch is the one with the lowest average SMAPE. A larger ``forecast_horizon`` normally lowers the
accuracy of the later time steps, which you can see in the per-time-step table of each run.

Available Models
----------------

The forecasting models enabled in ``tinyml_modelzoo/model_descriptions/forecasting.py`` are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model
     - Description
   * - ``FCST_LSTM8``, ``FCST_LSTM10``
     - One LSTM layer (hidden size 8 or 10) followed by a linear layer
   * - ``FCST_3k``
     - 4 fully connected layers, about 3k parameters
   * - ``FCST_13k``
     - 2 Conv, BatchNorm, ReLU layers and a linear layer, about 13k parameters
   * - ``FCST_500_NPU``, ``FCST_1k_NPU``, ``FCST_2k_NPU``, ``FCST_4k_NPU``, ``FCST_6k_NPU``, ``FCST_8k_NPU``,
       ``FCST_10k_NPU``, ``FCST_20k_NPU``
     - NPU-compliant convolutional models of about 0.5k to 20k parameters (``FCST_6k_NPU`` uses depthwise
       separable convolutions)

Use an ``_NPU`` model on devices with an NPU (F28P55, F28P65, MSPM0G5187) and see :doc:`/appendix/model_zoo_reference`.

.. warning::

   The quantized ``FCST_LSTM10`` model has shown a large accuracy drop in 1.5.0 runs (see
   :doc:`generic_forecasting`). Always compare the float and quantized results of your own run.

Forecasting versus Regression
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 37 38

   * - Aspect
     - Regression
     - Forecasting
   * - Output
     - Value for the current window
     - Future values (``forecast_horizon`` steps)
   * - Task type
     - ``generic_timeseries_regression``
     - ``generic_timeseries_forecasting``
   * - Metrics
     - RMSE, R2
     - SMAPE, R2 per time step
   * - Examples
     - :doc:`generic_regression`, :doc:`torque_measurement_regression`
     - :doc:`generic_forecasting`, :doc:`forecasting_pmsm_rotor`

Deployment
----------

See :doc:`/deploying_forecasting_models_from_modelmaker_to_device/readme`.
