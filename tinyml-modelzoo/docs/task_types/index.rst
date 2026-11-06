====================
Supported Task Types
====================

Tiny ML ModelZoo supports seven machine learning tasks, all optimized
for deployment on resource-constrained microcontrollers.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   timeseries_classification
   timeseries_regression
   timeseries_forecasting
   anomaly_detection
   image_classification
   audio_classification
   radar_classification
   /RADAR_PROCESSING_PIPELINE

Task Overview
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Task Type
     - Description
     - Example Applications
   * - :doc:`timeseries_classification`
     - Categorize time-series data into discrete classes
     - Arc fault detection, motor bearing fault, activity recognition
   * - :doc:`timeseries_regression`
     - Predict continuous values from time-series inputs
     - Torque estimation, speed prediction, load measurement
   * - :doc:`timeseries_forecasting`
     - Predict future values based on historical patterns
     - Temperature prediction, demand forecasting
   * - :doc:`anomaly_detection`
     - Identify abnormal patterns using autoencoders
     - Equipment health monitoring, predictive maintenance
   * - :doc:`image_classification`
     - Categorize images into classes
     - Visual inspection, digit recognition
   * - :doc:`audio_classification`
     - Categorize short audio clips (new in 1.5.0)
     - Keyword spotting, wake word, cough and glass break detection
   * - :doc:`radar_classification`
     - Categorize windows of radar point-cloud or range-profile features (new in 1.5.0; surface classification in 1.6.0)
     - Human pose and fall detection

Task Type Strings and Modules
-----------------------------

``target_module`` is inferred from ``task_type`` when you leave it out of the config.

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - ``task_type``
     - ``target_module``
     - Dataset folder
   * - ``generic_timeseries_classification`` (also ``arc_fault``, ``motor_fault``, ``ecg_classification``,
       ``blower_imbalance``, ``pir_detection``)
     - ``timeseries``
     - ``classes/``
   * - ``generic_timeseries_regression``
     - ``timeseries``
     - ``files/``
   * - ``generic_timeseries_forecasting``
     - ``timeseries``
     - ``files/``
   * - ``generic_timeseries_anomalydetection``
     - ``timeseries``
     - ``classes/Normal``, ``classes/Anomaly``
   * - ``image_classification``
     - ``vision``
     - ``classes/``
   * - ``audio_classification``
     - ``audio``
     - ``classes/``
   * - ``radar_point_cloud_classification``, ``radar_surface_classification``
     - ``radar``
     - ``classes/``

Dataset layouts are described in :doc:`/byod/index`. Early stopping is on by default for all tasks since
1.5.0 (:doc:`/features/early_stopping`).

Choosing the Right Task
-----------------------

**Classification vs. Regression**

* Use **Classification** when you need to assign labels: "Is this A, B, or C?"
* Use **Regression** when you need to predict a number: "What is the value of X?"

**Anomaly Detection vs. Classification**

* Use **Anomaly Detection** when you primarily have "normal" data for training.
  Although only normal data is used to train the autoencoder, some anomaly data
  is still needed for testing and threshold evaluation. If anomaly data is not
  yet available, you can still train and deploy the model, then collect anomaly
  data over time to evaluate performance later. See
  :doc:`anomaly_detection` for details on the different scenarios.
* Use **Classification** when you have labeled examples of all conditions (including faults)

**Regression vs. Forecasting**

* Use **Regression** to predict a value at the current time instant
* Use **Forecasting** to predict future values in the time series
