==============================
Fan Blade Fault Classification
==============================

Detect faults in BLDC fans from accelerometer data.

Overview
--------

This example demonstrates fault detection in brushless DC (BLDC) fans using
vibration data from accelerometers. It can identify various fault conditions
including blade damage, bearing wear, and imbalance.

**Application**: Cooling systems, industrial fans, computer hardware

**Task Type**: Time Series Classification

**Data Type**: Multivariate (accelerometer X, Y, Z axes)

Demo Setup
----------

.. figure:: /_static/img/examples/fan_blade_fault/demo_setup.jpg
   :width: 600px
   :align: center
   :alt: Fan Blade Fault Demo Setup

   Hardware setup for fan blade fault classification demo

Fault Types
-----------

The model can identify various fault conditions:

.. list-table::
   :widths: 25 75

   * - .. figure:: /_static/img/examples/fan_blade_fault/normal.jpg
          :width: 200px
          :alt: Normal Operation
     - **Normal Operation** - Fan running without any faults

   * - .. figure:: /_static/img/examples/fan_blade_fault/blade_damage.jpg
          :width: 200px
          :alt: Blade Damage
     - **Blade Damage** - Physical damage to fan blades

   * - .. figure:: /_static/img/examples/fan_blade_fault/blade_imbalance.jpg
          :width: 200px
          :alt: Blade Imbalance
     - **Blade Imbalance** - Uneven weight distribution causing vibration

   * - .. figure:: /_static/img/examples/fan_blade_fault/blade_obstruction.jpg
          :width: 200px
          :alt: Blade Obstruction
     - **Blade Obstruction** - Foreign object interfering with fan operation

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'fan_blade_fault_classification'

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
         ./run_tinyml_modelzoo.sh examples/fan_blade_fault_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\fan_blade_fault_classification\config.yaml

Dataset Details
---------------

**Input Variables**:

* Accelerometer X-axis
* Accelerometer Y-axis
* Accelerometer Z-axis

**Classes**:

* Normal operation
* Blade fault
* Bearing fault
* Imbalance

Results and Analysis
--------------------

**ROC Curves:**

.. figure:: /_static/img/examples/fan_blade_fault/One_vs_Rest_MultiClass_ROC_test.png
   :width: 600px
   :align: center
   :alt: ROC Curves

   One-vs-Rest Multi-class ROC curves showing excellent classification performance

**Class Score Histogram:**

.. figure:: /_static/img/examples/fan_blade_fault/Histogram_Class_Score_differences_test.png
   :width: 600px
   :align: center
   :alt: Class Score Histogram

   Distribution of class score differences

**Feature Extraction Quality:**

.. figure:: /_static/img/examples/fan_blade_fault/pca_on_feature_extracted_train_data.png
   :width: 600px
   :align: center
   :alt: PCA on Training Data

   PCA visualization showing class separation in feature space

Anomaly Detection Variant
-------------------------

This example also supports anomaly detection mode:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/fan_blade_fault_classification/config_anomaly_detection.yaml

**Reconstruction Error Analysis:**

.. figure:: /_static/img/examples/fan_blade_fault/reconstruction_error_log_scale.png
   :width: 600px
   :align: center
   :alt: Reconstruction Error

   Reconstruction error distribution for anomaly detection

See Also
--------

* :doc:`blower_imbalance` - Current-based imbalance detection
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/fan_blade_fault_classification>`_
