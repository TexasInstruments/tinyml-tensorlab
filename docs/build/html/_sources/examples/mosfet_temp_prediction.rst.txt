=========================================
MOSFET Junction Temperature Prediction
=========================================

Predict MOSFET junction temperature for thermal management in power conversion systems.

Overview
--------

High voltage switches are used widely in power conversion and motor drive
applications. Accurate temperature prediction of these switches is critical for
protecting the switching devices while delivering the best system efficiency.
However, direct measurement of the switch temperature is often difficult and
expensive.

This example provides a generic approach for predicting switch temperature
based on the power loss of a switch and indirect temperature measurement from
a temperature sensor in the vicinity of the switch.

The approach uses a linear + AI modeling method:

* **Linear model**: A linear autoregressive-moving-average (ARMA) model describes
  the linear thermal behavior from power loss to the difference between case
  temperature and NTC sensor temperature.
* **AI model**: An MLP model captures the residual nonlinear error between the
  linear model prediction and true measurement.

This combination provides more accurate models, more constrained AI behavior,
and a more robust overall prediction.

**Application**: Thermal management in power converters, motor drives, EV power electronics

**Task Type**: Time Series Regression

**Data Type**: Multivariate (43 input variables)

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_regression'
     target_device: 'F29H85'

   dataset:
     dataset_name: 'mosfet_temp_prediction'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/mosfet_temp_prediction.zip'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     variables: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
     stride_size: 1
     frame_size: 1

   training:
     model_name: 'REGR_3k'
     batch_size: 128
     training_epochs: 100
     learning_rate: 0.01
     quantization: 0
     output_int: false

   testing: {}
   compilation: {}

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/mosfet_temp_prediction/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\mosfet_temp_prediction\config.yaml

Dataset Details
---------------

**Input Variables (43 total)**:

* Past NTC temperature data (20 time points)
* Past power loss data (20 time points)
* Meta information (ambient temperature, coolant temperature, coolant flow rate)

The dataset contains temperature data for various ambient temperatures, coolant
temperatures, and coolant flow rate configurations.

**Output**:

* T_case - T_linear (difference between expected case temperature and linear
  model output)

**Dataset Download**:

The dataset is automatically downloaded from:

.. code-block:: yaml

   dataset:
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/mosfet_temp_prediction.zip'

On-Device Deployment
--------------------

This example targets the F29H85x MCU. After running ModelMaker, copy the
compiled model files to the CCS example project:

* ``mod.a`` - The compiled model library
* ``tvmgen_default.h`` - Header file for model inference APIs

The CCS example ``generic_timeseries_regression`` for F29H85x provides the
application framework.

See Also
--------

* :doc:`torque_measurement_regression` - Motor torque prediction
* :doc:`induction_motor_speed_prediction` - Motor speed prediction
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/mosfet_temp_prediction>`_
