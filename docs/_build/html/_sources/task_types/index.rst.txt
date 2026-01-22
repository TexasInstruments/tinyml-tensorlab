====================
Supported Task Types
====================

Tiny ML Tensorlab supports five primary machine learning tasks, all optimized
for deployment on resource-constrained microcontrollers.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   timeseries_classification
   timeseries_regression
   timeseries_forecasting
   anomaly_detection
   image_classification

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

Choosing the Right Task
-----------------------

**Classification vs. Regression**

* Use **Classification** when you need to assign labels: "Is this A, B, or C?"
* Use **Regression** when you need to predict a number: "What is the value of X?"

**Anomaly Detection vs. Classification**

* Use **Anomaly Detection** when you only have "normal" data for training
* Use **Classification** when you have labeled examples of all conditions (including faults)

**Regression vs. Forecasting**

* Use **Regression** to predict a value at the current time instant
* Use **Forecasting** to predict future values in the time series
