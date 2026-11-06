=======================
Examples & Applications
=======================

``tinyml-modelzoo`` ships ready-to-run examples under ``examples/``. Each example folder has one or more
config YAML files (one per target device or variant), and most have a ``readme.md``. This section has one
page per example, grouped by task.

.. note::

   Many example folders have no plain ``config.yaml``. Use the config file names listed on each page,
   for example ``examples/dc_arc_fault/config_dsk.yaml``.

Running an Example
------------------

From the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows (cmd)

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         ./run_tinyml_modelzoo.ps1 examples/generic_timeseries_classification/config.yaml

Results are written to ``./data/projects/<project_name>/run/<date-time>/<model_name>/``; see
:doc:`/getting_started/understanding_run_output`. If you installed the wheel only, see
:doc:`/installation/user_installation` for how to get the example configs. To adapt an example, see
:doc:`/CONFIG_YAML_GUIDE`. To run a trained example on a board, see :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

.. important::

   Result numbers on the example pages come from the example ``readme.md`` files (written for earlier
   releases) or from short automated test runs of 1.5.0, as labelled on each page. They show what to
   expect, not a benchmark.

Generic Examples (Start Here)
-----------------------------

One per timeseries task. Use these to learn the flow, then point the config at your own data.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Description
   * - :doc:`generic_classification`
     - Hello World: classify sine, square and sawtooth waveforms.
   * - :doc:`generic_regression`
     - Hello World regression on synthetic data.
   * - :doc:`generic_forecasting`
     - Hello World forecasting on a simulated thermostat signal.
   * - :doc:`generic_anomaly_detection`
     - Autoencoder anomaly detection on synthetic data.

Timeseries Classification
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Description
   * - :doc:`arc_fault`
     - DC arc fault detection from current waveforms (classification, anomaly detection, on-device learning).
   * - :doc:`ac_arc_fault`
     - AC arc fault detection on MSPM0G5187.
   * - :doc:`motor_bearing_fault`
     - Motor bearing fault from 3-axis vibration (classification and anomaly detection).
   * - :doc:`blower_imbalance`
     - Blower imbalance from a 3-variable signal.
   * - :doc:`fan_blade_fault_classification`
     - Fan blade faults from an ADXL355 accelerometer, including wireless MCUs and on-device training.
   * - :doc:`electrical_fault`
     - Transmission line fault classification (2-class and 6-class).
   * - :doc:`grid_stability`
     - Smart grid stable vs unstable (UCI 4-node star grid).
   * - :doc:`grid_fault_detection`
     - Single-phase on-board-charger grid faults on F29H85.
   * - :doc:`gas_sensor`
     - Gas identification from 10 MOS sensors.
   * - :doc:`har_activity_recognition`
     - Human activity recognition (WISDM) with a residual model.
   * - :doc:`ecg_classification`
     - ECG classification (classification and anomaly detection).
   * - :doc:`nilm_classification`
     - Appliance usage classification (ESDA NILM and PLAID).
   * - :doc:`pir_detection`
     - Motion detection from a PIR sensor on MSPM0 and wireless MCUs.
   * - :doc:`gearbox_fault_detection`
     - Gearbox fault from 4-channel vibration.
   * - :doc:`fall_detection_classification`
     - Fall detection from an accelerometer (SisFall).
   * - :doc:`dynamic_hand_gesture_recognition`
     - Hand gestures from a 3-axis accelerometer.
   * - :doc:`wifi_csi_presence_detection`
     - Presence detection from Wi-Fi CSI on CC35X1 (new in 1.5.0).

Timeseries Regression
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Description
   * - :doc:`torque_measurement_regression`
     - PMSM torque estimation.
   * - :doc:`induction_motor_speed_prediction`
     - Induction motor speed (RPM) prediction.
   * - :doc:`washing_machine_regression`
     - Laundry load weighing from motor signals.
   * - :doc:`mosfet_temp_prediction`
     - MOSFET case-temperature prediction (linear model plus AI residual).
   * - :doc:`bms_soc_estimation`
     - Battery state-of-charge estimation (new in 1.5.0).

Timeseries Forecasting
----------------------

:doc:`forecasting_example` lists all forecasting configs, the windowing rules and the ``FCST_*`` models.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Description
   * - :doc:`forecasting_pmsm_rotor`
     - PMSM rotor magnet temperature, one step ahead.
   * - :doc:`hvac_indoor_temp_forecast`
     - Indoor temperature forecasting for HVAC.

Anomaly Detection
-----------------

Anomaly detection configs live inside several example folders. :doc:`anomaly_detection_example` lists all of
them with thresholds and troubleshooting.

Audio Classification
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Description
   * - :doc:`google_speech_command`
     - 12-class keyword spotting with MFCC features.
   * - :doc:`wake_word_detection`
     - Wake word detection with a learned filterbank (new in 1.5.0).
   * - :doc:`glass_break_detection`
     - Glass break detection (new in 1.5.0).
   * - :doc:`cough_detection`
     - Cough detection (new in 1.5.0).

Image Classification
--------------------

:doc:`image_classification_example` is an overview of the image examples.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Description
   * - :doc:`mnist_image_classification`
     - Digit recognition with LeNet-5.
   * - :doc:`machine_readable_code_classification`
     - QR code vs barcode vs other.
   * - :doc:`coffee_bean_classification`
     - Coffee roast level classification.

Radar
-----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Description
   * - :doc:`radar_posture_and_fall_detection`
     - Posture and fall classification from radar point cloud and track data (1.5.0, reworked in 1.6.0).
   * - :doc:`radar_surface_classification`
     - Surface/material classification from FFT range profiles (new in 1.6.0).

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
   grid_fault_detection
   gas_sensor
   har_activity_recognition
   ecg_classification
   nilm_classification
   pir_detection
   gearbox_fault_detection
   fall_detection_classification
   dynamic_hand_gesture_recognition
   wifi_csi_presence_detection
   torque_measurement_regression
   induction_motor_speed_prediction
   washing_machine_regression
   mosfet_temp_prediction
   bms_soc_estimation
   forecasting_pmsm_rotor
   hvac_indoor_temp_forecast
   forecasting_example
   anomaly_detection_example
   google_speech_command
   wake_word_detection
   glass_break_detection
   cough_detection
   mnist_image_classification
   machine_readable_code_classification
   coffee_bean_classification
   image_classification_example
   radar_posture_and_fall_detection
   radar_surface_classification
