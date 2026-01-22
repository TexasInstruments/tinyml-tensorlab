==========================
Washing Machine Regression
==========================

Predict washing machine load weight from sensor data.

Overview
--------

This example demonstrates load weight estimation for washing machines using
motor current and vibration data. Accurate load estimation enables optimal
water and detergent dosing, improving efficiency and wash quality.

**Application**: Smart appliances, home automation, energy efficiency

**Task Type**: Time Series Regression

**Data Type**: Multivariate (motor and sensor data)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_regression'
     target_device: 'MSPM0G5187'

   dataset:
     dataset_name: 'reg_washing_machine'

   training:
     model_name: 'REGR_2k_NPU'
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
         ./run_tinyml_modelzoo.sh examples/reg_washing_machine/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\reg_washing_machine\config.yaml

Dataset Details
---------------

**Input Variables**:

* Motor current
* Drum speed
* Vibration sensor data

**Output**:

* Load weight (continuous value in kg)

Results
-------

**Float Model Predictions:**

.. figure:: /_static/img/examples/washing_machine/float_actual_vs_predicted.png
   :width: 600px
   :align: center
   :alt: Float Model Predictions

   Actual vs predicted load weight using float model

**Quantized Model Predictions:**

.. figure:: /_static/img/examples/washing_machine/partially_quantized_actual_vs_predicted.png
   :width: 600px
   :align: center
   :alt: Quantized Model Predictions

   Actual vs predicted load weight using quantized model

See Also
--------

* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/reg_washing_machine>`_
