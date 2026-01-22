===========================
Human Activity Recognition
===========================

Classify human activities from accelerometer and gyroscope data.

Overview
--------

This example demonstrates Human Activity Recognition (HAR) using inertial
sensor data. It classifies activities such as walking, running, sitting,
and standing based on accelerometer and gyroscope readings.

**Application**: Wearables, fitness trackers, smart home, elderly care

**Task Type**: Time Series Classification

**Data Type**: Multivariate (accelerometer + gyroscope)

.. note::
   This example uses a branched model architecture for improved accuracy.

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'branched_model_parameters'

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
         ./run_tinyml_modelzoo.sh examples/branched_model_parameters/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\branched_model_parameters\config.yaml

Dataset Details
---------------

**Input Variables**:

* Accelerometer X, Y, Z
* Gyroscope X, Y, Z

**Activity Classes**:

* Walking
* Running
* Sitting
* Standing
* Lying down
* Walking upstairs
* Walking downstairs

See Also
--------

* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/branched_model_parameters>`_
