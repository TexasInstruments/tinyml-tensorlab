================
Blower Imbalance
================

Detect blade imbalance in HVAC blowers using 3-phase motor current analysis.

Overview
--------

Blade imbalance in HVAC blowers causes increased vibration, noise, and
premature bearing wear. This example uses motor current signature analysis
to detect imbalance conditions before they cause equipment failure.

**Application**: HVAC systems, industrial fans, predictive maintenance

**Task Type**: Time Series Classification

**Data Type**: Multivariate (3-phase motor currents)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'blower_imbalance'

   training:
     model_name: 'FanImbalance_model_1_t'
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
         ./run_tinyml_modelzoo.sh examples/blower_imbalance/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\blower_imbalance\config.yaml

Dataset Details
---------------

**Input Variables**:

* Phase A current
* Phase B current
* Phase C current

**Classes**:

* Normal operation
* Blade imbalance detected

Recommended Models
------------------

.. important::

   The ``FanImbalance_model_*`` models listed below are only available in
   TI's **Edge AI Studio** (GUI) and are not included in Tensorlab.
   Use the generic ``CLS_*_NPU`` models (e.g., ``CLS_1k_NPU``,
   ``CLS_4k_NPU``) as equivalent alternatives in Tensorlab.

**Edge AI Studio models:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Parameters
     - Use Case
   * - ``FanImbalance_model_1_t``
     - Varies
     - Baseline detection
   * - ``FanImbalance_model_2_t``
     - Varies
     - Improved accuracy
   * - ``FanImbalance_model_3_t``
     - Varies
     - Maximum accuracy

**Tensorlab alternatives**: Use ``CLS_1k_NPU``, ``CLS_2k_NPU``, or
``CLS_4k_NPU`` for equivalent performance.

See Also
--------

* :doc:`fan_blade_fault_classification` - Accelerometer-based fan fault detection
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/blower_imbalance>`_
