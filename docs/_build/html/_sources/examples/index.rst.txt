=======================
Examples & Applications
=======================

This section provides ready-to-run examples demonstrating various AI applications
for TI microcontrollers. Each example includes complete configuration files and
step-by-step instructions.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   hello_world
   arc_fault
   ac_arc_fault
   motor_bearing_fault
   blower_imbalance
   fan_blade_fault_classification
   electrical_fault
   grid_stability
   gas_sensor
   har_activity_recognition
   ecg_classification
   nilm_classification
   pir_detection
   torque_measurement_regression
   induction_motor_speed_prediction
   washing_machine_regression
   forecasting_pmsm_rotor
   hvac_indoor_temp_forecast
   anomaly_detection_example
   forecasting_example
   mnist_image_classification
   image_classification_example

Running an Example
------------------

All examples are located in ``tinyml-modelzoo/examples/``. To run an example:

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/<example_name>/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\<example_name>\config.yaml

Output artifacts will be saved to ``../tinyml-modelmaker/data/projects/<project_name>/``.

----

Classification Examples
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 5 25 15 55

   * - No.
     - Example
     - Data Type
     - Description
   * - 1
     - `hello_world <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/hello_world>`_
     - Univariate
     - Classify sine/square/sawtooth waveforms. **Start here** to learn the toolchain.
   * - 2
     - `dc_arc_fault <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/dc_arc_fault>`_
     - Univariate
     - Detect DC arc faults from current waveforms for electrical safety.
   * - 3
     - `ac_arc_fault <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ac_arc_fault>`_
     - Univariate
     - Detect AC arc faults in electrical systems.
   * - 4
     - `motor_bearing_fault <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/motor_bearing_fault>`_
     - Multivariate
     - Classify 5 bearing fault types + normal operation from vibration data.
   * - 5
     - `blower_imbalance <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/blower_imbalance>`_
     - Multivariate
     - Detect blade imbalance in HVAC blowers using 3-phase motor currents.
   * - 6
     - `fan_blade_fault_classification <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/fan_blade_fault_classification>`_
     - Multivariate
     - Detect faults in BLDC fans from accelerometer data.
   * - 7
     - `electrical_fault <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/electrical_fault>`_
     - Multivariate
     - Classify transmission line faults using voltage and current.
   * - 8
     - `grid_stability <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/grid_stability>`_
     - Multivariate
     - Predict power grid stability from node parameters.
   * - 9
     - `gas_sensor <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/gas_sensor>`_
     - Multivariate
     - Identify gas type and concentration from sensor array data.
   * - 10
     - `branched_model_parameters <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/branched_model_parameters>`_
     - Multivariate
     - Human Activity Recognition from accelerometer/gyroscope data.
   * - 11
     - `ecg_classification <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ecg_classification>`_
     - Multivariate
     - Classify normal vs anomalous heartbeats from ECG signals.
   * - 12
     - `nilm_appliance_usage_classification <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/nilm_appliance_usage_classification>`_
     - Multivariate
     - Non-Intrusive Load Monitoring - identify active appliances.
   * - 13
     - `PLAID_nilm_classification <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/PLAID_nilm_classification>`_
     - Multivariate
     - Appliance identification using the PLAID dataset.
   * - 14
     - `pir_detection <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/pir_detection>`_
     - Multivariate
     - Detect presence/motion using PIR sensor data.

----

Regression Examples
-------------------

.. list-table::
   :header-rows: 1
   :widths: 5 25 15 55

   * - No.
     - Example
     - Data Type
     - Description
   * - 1
     - `torque_measurement_regression <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/torque_measurement_regression>`_
     - Multivariate
     - Predict PMSM motor torque from current measurements.
   * - 2
     - `induction_motor_speed_prediction <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/induction_motor_speed_prediction>`_
     - Multivariate
     - Predict induction motor speed from electrical signals.
   * - 3
     - `reg_washing_machine <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/reg_washing_machine>`_
     - Multivariate
     - Predict washing machine load weight.

----

Forecasting Examples
--------------------

.. list-table::
   :header-rows: 1
   :widths: 5 25 15 55

   * - No.
     - Example
     - Data Type
     - Description
   * - 1
     - `forecasting_pmsm_rotor <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/forecasting_pmsm_rotor>`_
     - Multivariate
     - Forecast PMSM rotor winding temperature.
   * - 2
     - `hvac_indoor_temp_forecast <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/hvac_indoor_temp_forecast>`_
     - Multivariate
     - Predict indoor temperature for HVAC control.

----

Anomaly Detection Examples
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 5 25 15 55

   * - No.
     - Example
     - Data Type
     - Description
   * - 1
     - `dc_arc_fault (DSI) <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/dc_arc_fault>`_
     - Univariate
     - Detect anomalous DC arc patterns using autoencoder (DSI dataset).
   * - 2
     - `dc_arc_fault (DSK) <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/dc_arc_fault>`_
     - Univariate
     - Detect anomalous DC arc patterns using autoencoder (DSK dataset).
   * - 3
     - `ecg_classification <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ecg_classification>`_
     - Multivariate
     - Detect anomalous heartbeat patterns from ECG signals.
   * - 4
     - `fan_blade_fault_classification <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/fan_blade_fault_classification>`_
     - Multivariate
     - Detect anomalous fan blade behavior from accelerometer data.
   * - 5
     - `motor_bearing_fault <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/motor_bearing_fault>`_
     - Multivariate
     - Detect anomalous bearing behavior from vibration data.

----

Image Classification Examples
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 5 25 15 55

   * - No.
     - Example
     - Data Type
     - Description
   * - 1
     - `MNIST_image_classification <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/MNIST_image_classification>`_
     - Image
     - Handwritten digit recognition (MNIST dataset).

----

Generic Examples
----------------

If you do not find an application that matches your use case, use these generic
examples as starting points:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Task Type
     - Example
     - Description
   * - Time Series Classification
     - :doc:`hello_world`
     - **Start here** to learn the toolchain
   * - Time Series Regression
     - `torque_measurement_regression <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/torque_measurement_regression>`_
     - Adapt for your regression task
   * - Time Series Forecasting
     - :doc:`forecasting_example`
     - Forecasting tutorial
   * - Anomaly Detection
     - :doc:`anomaly_detection_example`
     - Autoencoder-based detection tutorial

----

Detailed Example Walkthroughs
-----------------------------

The following pages provide detailed walkthroughs for selected examples:

* :doc:`hello_world` - Step-by-step introduction to the toolchain
* :doc:`arc_fault` - DC/AC arc fault detection with industrial applications
* :doc:`motor_bearing_fault` - Vibration-based fault classification
* :doc:`anomaly_detection_example` - Autoencoder-based anomaly detection
* :doc:`forecasting_example` - Time series forecasting techniques
* :doc:`image_classification_example` - Image classification on MCUs
