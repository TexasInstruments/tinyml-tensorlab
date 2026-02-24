==========================
HVAC Indoor Temp Forecast
==========================

Predict indoor temperature for HVAC control.

Overview
--------

This example demonstrates indoor temperature forecasting for HVAC systems.
Predicting future temperatures enables predictive control strategies that
improve comfort and energy efficiency.

**Application**: Building automation, smart thermostats, energy management

**Task Type**: Time Series Forecasting

**Data Type**: Multivariate (environmental and HVAC parameters)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_forecasting'
     target_device: 'MSPM0G5187'

   dataset:
     dataset_name: 'hvac_indoor_temp_forecast'

   training:
     model_name: 'FCST_4k_NPU'
     training_epochs: 50
     batch_size: 32

   testing: {}
   compilation: {}

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/hvac_indoor_temp_forecast/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\hvac_indoor_temp_forecast\config.yaml

Dataset Details
---------------

**Input Variables**:

* Current indoor temperature
* Outdoor temperature
* HVAC system state
* Occupancy (optional)
* Time of day
* Historical temperature data

**Forecast Target**:

* Future indoor temperature

Results
-------

**Temperature Prediction Plot:**

.. figure:: /_static/img/examples/hvac_indoor_temp/indoorTemperature_predictions.png
   :width: 700px
   :align: center
   :alt: Indoor Temperature Predictions

   Forecast vs actual indoor temperature showing prediction accuracy

**On-Device Results:**

.. figure:: /_static/img/examples/hvac_indoor_temp/ondevice_results.png
   :width: 600px
   :align: center
   :alt: On-Device Results

   Inference results on target device

See Also
--------

* :doc:`forecasting_pmsm_rotor` - Motor temperature forecasting
* :doc:`forecasting_example` - Forecasting tutorial
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/hvac_indoor_temp_forecast>`_
