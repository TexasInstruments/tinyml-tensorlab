==========
Gas Sensor
==========

Identify gas type and concentration from sensor array data.

Overview
--------

This example demonstrates gas identification using data from a sensor array.
It can classify different gas types and estimate concentrations based on
the response patterns of multiple gas sensors.

**Application**: Environmental monitoring, industrial safety, air quality

**Task Type**: Time Series Classification

**Data Type**: Multivariate (multiple gas sensor readings)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'gas_sensor'

   training:
     model_name: 'CLS_4k_NPU'
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
         ./run_tinyml_modelzoo.sh examples/gas_sensor/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\gas_sensor\config.yaml

Dataset Details
---------------

**Input Variables**:

* Multiple gas sensor readings (MOX sensors)
* Temperature and humidity (optional)

**Classes**:

* Different gas types (e.g., CO, NO2, Ethanol, etc.)
* Concentration levels

Quantization Analysis
---------------------

.. figure:: /_static/img/examples/gas_sensor/quantize_vs_dequantize.png
   :width: 600px
   :align: center
   :alt: Quantization Comparison

   Comparison of quantized vs dequantized model outputs

See Also
--------

* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/gas_sensor>`_
