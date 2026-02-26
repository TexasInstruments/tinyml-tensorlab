=======================
Examples & Applications
=======================

This section provides ready-to-run examples demonstrating various AI applications
for TI microcontrollers. Each example includes complete configuration files and
step-by-step instructions.

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

You can choose to save the output artifacts in your own custom directory by specifying
in the respective ``config.yaml`` under the common section:

.. code-block:: yaml

   common:
     projects_path: './your/choice'  # or absolute path
     # ... other settings

----

Generic Examples
----------------

If you do not find an application that matches your use case, use these generic
examples as starting points:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Task Type
     - Example
     - Description
   * - Time Series Classification
     - :doc:`generic_classification` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/generic_timeseries_classification>`__)
     - Classify sine/square/sawtooth waveforms. **Start here** to learn the toolchain.
   * - Time Series Regression
     - :doc:`generic_regression` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/generic_timseries_regression>`__)
     - Generic regression example for continuous value prediction.
   * - Time Series Forecasting
     - :doc:`generic_forecasting` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/generic_timeseries_forecasting>`__)
     - Generic forecasting example for time series prediction.
   * - Time Series Anomaly Detection
     - :doc:`generic_anomaly_detection` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/generic_timeseries_anomalydetection>`__)
     - Generic anomaly detection example using autoencoders.

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
     - :doc:`arc_fault` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/dc_arc_fault>`__)
     - Univariate
     - Detect DC arc faults from current waveforms for electrical safety.
   * - 2
     - :doc:`ac_arc_fault` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ac_arc_fault>`__)
     - Univariate
     - Detect AC arc faults in electrical systems.
   * - 3
     - :doc:`motor_bearing_fault` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/motor_bearing_fault>`__)
     - Multivariate
     - Classify 5 bearing fault types + normal operation from vibration data.
   * - 4
     - :doc:`blower_imbalance` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/blower_imbalance>`__)
     - Multivariate
     - Detect blade imbalance in HVAC blowers using 3-phase motor currents.
   * - 5
     - :doc:`fan_blade_fault_classification` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/fan_blade_fault_classification>`__)
     - Multivariate
     - Detect faults in BLDC fans from accelerometer data.
   * - 6
     - :doc:`electrical_fault` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/electrical_fault>`__)
     - Multivariate
     - Classify transmission line faults using voltage and current.
   * - 7
     - :doc:`grid_stability` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/grid_stability>`__)
     - Multivariate
     - Predict power grid stability from node parameters.
   * - 8
     - :doc:`gas_sensor` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/gas_sensor>`__)
     - Multivariate
     - Identify gas type and concentration from sensor array data.
   * - 9
     - :doc:`har_activity_recognition` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/branched_model_parameters>`__)
     - Multivariate
     - Human Activity Recognition from accelerometer/gyroscope data.
   * - 10
     - :doc:`ecg_classification` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ecg_classification>`__)
     - Multivariate
     - Classify normal vs anomalous heartbeats from ECG signals.
   * - 11
     - :doc:`nilm_classification` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/nilm_appliance_usage_classification>`__)
     - Multivariate
     - Non-Intrusive Load Monitoring - identify active appliances.
   * - 12
     - :doc:`pir_detection` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/pir_detection>`__)
     - Multivariate
     - Detect presence/motion using PIR sensor data.
   * - 13
     - :doc:`grid_fault_detection` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/grid_fault_detection>`__)
     - Univariate
     - Detect AC grid faults in EV on-board chargers using current measurements.

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
     - :doc:`torque_measurement_regression` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/torque_measurement_regression>`__)
     - Multivariate
     - Predict PMSM motor torque from current measurements.
   * - 2
     - :doc:`induction_motor_speed_prediction` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/induction_motor_speed_prediction>`__)
     - Multivariate
     - Predict induction motor speed from electrical signals.
   * - 3
     - :doc:`washing_machine_regression` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/reg_washing_machine>`__)
     - Multivariate
     - Predict washing machine load weight.
   * - 4
     - :doc:`mosfet_temp_prediction` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/mosfet_temp_prediction>`__)
     - Multivariate
     - Predict MOSFET junction temperature for thermal management in power converters.

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
     - :doc:`forecasting_pmsm_rotor` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/forecasting_pmsm_rotor>`__)
     - Multivariate
     - Forecast PMSM rotor winding temperature.
   * - 2
     - :doc:`hvac_indoor_temp_forecast` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/hvac_indoor_temp_forecast>`__)
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
     - :doc:`anomaly_detection_example` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/dc_arc_fault>`__)
     - Univariate
     - Arc fault anomaly detection using autoencoder.
   * - 2
     - :doc:`ecg_classification` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ecg_classification>`__)
     - Multivariate
     - Detect anomalous heartbeat patterns from ECG signals.
   * - 3
     - :doc:`fan_blade_fault_classification` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/fan_blade_fault_classification>`__)
     - Multivariate
     - Detect anomalous fan blade behavior from accelerometer data.
   * - 4
     - :doc:`motor_bearing_fault` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/motor_bearing_fault>`__)
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
     - :doc:`mnist_image_classification` (`GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/MNIST_image_classification>`__)
     - Image
     - Handwritten digit recognition (MNIST dataset).

----

.. toctree::
   :hidden:
   :maxdepth: 1

   generic_classification
   generic_regression
   generic_forecasting
   generic_anomaly_detection
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
   grid_fault_detection
   torque_measurement_regression
   induction_motor_speed_prediction
   washing_machine_regression
   mosfet_temp_prediction
   forecasting_pmsm_rotor
   hvac_indoor_temp_forecast
   anomaly_detection_example
   forecasting_example
   image_classification_example
   mnist_image_classification
