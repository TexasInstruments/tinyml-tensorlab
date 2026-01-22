=============================
Torque Measurement Regression
=============================

Predict PMSM motor torque from current measurements.

Overview
--------

This example demonstrates torque estimation for Permanent Magnet Synchronous
Motors (PMSM) using phase current measurements. Accurate torque estimation
enables sensorless torque control, reducing cost and complexity.

**Application**: Motor drives, electric vehicles, industrial automation

**Task Type**: Time Series Regression

**Data Type**: Multivariate (motor currents)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_regression'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'torque_measurement_regression'

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
         ./run_tinyml_modelzoo.sh examples/torque_measurement_regression/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\torque_measurement_regression\config.yaml

Dataset Details
---------------

**Input Variables**:

* Phase A current (Ia)
* Phase B current (Ib)
* Phase C current (Ic)
* Rotor position (optional)

**Output**:

* Torque (continuous value in Nm)

Recommended Models
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Parameters
     - Use Case
   * - ``REGR_1k_NPU``
     - ~1,000
     - Basic estimation
   * - ``REGR_4k_NPU``
     - ~4,000
     - Balanced accuracy
   * - ``REGR_8k_NPU``
     - ~8,000
     - High accuracy

See Also
--------

* :doc:`induction_motor_speed_prediction` - Speed prediction
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/torque_measurement_regression>`_
