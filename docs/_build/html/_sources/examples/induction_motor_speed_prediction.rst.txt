================================
Induction Motor Speed Prediction
================================

Predict induction motor speed from electrical signals.

Overview
--------

This example demonstrates speed estimation for induction motors using
electrical measurements. Sensorless speed estimation eliminates the need
for mechanical speed sensors, reducing cost and improving reliability.

**Application**: Industrial motor drives, pumps, fans, compressors

**Task Type**: Time Series Regression

**Data Type**: Multivariate (voltage and current signals)

**Induction Motor Relationships:**

.. figure:: /_static/img/examples/induction_motor/induction_motor_relationships.png
   :width: 600px
   :align: center
   :alt: Induction Motor Relationships

   Relationships between electrical parameters and motor speed

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_regression'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'induction_motor_speed_prediction'

   training:
     model_name: 'REGR_4k_NPU'
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
         ./run_tinyml_modelzoo.sh examples/induction_motor_speed_prediction/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\induction_motor_speed_prediction\config.yaml

Dataset Details
---------------

**Input Variables**:

* Phase voltages
* Phase currents
* DC bus voltage (optional)

**Output**:

* Motor speed (continuous value in RPM or rad/s)

See Also
--------

* :doc:`torque_measurement_regression` - Torque estimation
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/induction_motor_speed_prediction>`_
