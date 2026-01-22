======================
PMSM Rotor Forecasting
======================

Forecast PMSM rotor winding temperature.

Overview
--------

This example demonstrates temperature forecasting for PMSM motor rotor
windings. Predicting temperature rise helps prevent overheating and enables
proactive thermal management in motor drive applications.

**Application**: Motor thermal management, electric vehicles, industrial drives

**Task Type**: Time Series Forecasting

**Data Type**: Multivariate (motor operating parameters)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_forecasting'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'forecasting_pmsm_rotor'

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
         ./run_tinyml_modelzoo.sh examples/forecasting_pmsm_rotor/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\forecasting_pmsm_rotor\config.yaml

Dataset Details
---------------

**Input Variables**:

* Motor current
* Motor speed
* Ambient temperature
* Coolant temperature
* Historical rotor temperature

**Forecast Target**:

* Future rotor winding temperature

Recommended Models
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Parameters
     - Use Case
   * - ``FCST_2k_NPU``
     - ~2,000
     - Basic forecasting
   * - ``FCST_4k_NPU``
     - ~4,000
     - Balanced accuracy
   * - ``FCST_LSTM8``
     - Varies
     - LSTM-based (non-NPU)

Results
-------

**Prediction Plot:**

.. figure:: /_static/img/examples/forecasting_pmsm_rotor/pm_predictions.png
   :width: 700px
   :align: center
   :alt: PMSM Temperature Predictions

   Forecast vs actual rotor temperature showing prediction accuracy

**On-Device Results:**

.. figure:: /_static/img/examples/forecasting_pmsm_rotor/ondevice_results.png
   :width: 600px
   :align: center
   :alt: On-Device Results

   Inference results on target device

See Also
--------

* :doc:`hvac_indoor_temp_forecast` - HVAC temperature forecasting
* :doc:`forecasting_example` - Forecasting tutorial
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/forecasting_pmsm_rotor>`_
