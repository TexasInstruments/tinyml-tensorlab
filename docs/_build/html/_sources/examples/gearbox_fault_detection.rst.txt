=======================
Gearbox Fault Detection
=======================

Classify gearbox operating conditions from vibration sensor data.

Overview
--------

This example demonstrates real-time gearbox fault classification using 4-channel
vibration data on the MSPM0G5187 MCU with integrated NPU. The model detects healthy
operation vs. broken tooth faults, enabling predictive maintenance on
resource-constrained microcontrollers without cloud connectivity.

**Application**: Predictive maintenance, industrial machinery, gearbox health monitoring

**Task Type**: Time Series Classification

**Data Type**: Multivariate (4 vibration sensors)

Device Support
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Device
     - Description
     - Configuration File
   * - ``MSPM0G5187``
     - MSPM0 with NPU (primary)
     - ``config_MSPM0.yaml``

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ with integrated NPU

**Software**

* Code Composer Studio 12.x or later
* MSPM0 SDK 2.10.00 or later
* TI Edge AI Studio

Dataset
-------

The example uses the **SpectraQuest Gearbox Fault Diagnostics Simulator** dataset:

* **Source**: `Gearbox Fault Diagnosis — Kaggle <https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis>`_
* **Classes**: 2 (Healthy, Broken Tooth)
* **Operating conditions**: 10 load levels (0–90%)

Each CSV file contains 4 accelerometer columns (``a1``, ``a2``, ``a3``, ``a4``)
with ~88,000 rows per file. File naming convention: ``{b|h}30hz{load}.csv``
where ``b`` = broken tooth, ``h`` = healthy.

Feature Extraction
------------------

1. **Sensor Input** — 4-channel accelerometer data
2. **Windowing** — 256-sample fixed-length windows
3. **Single frame** passed directly to model

Models
------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Model
     - Parameters
     - Flash (bytes)
     - Description
   * - ``GearboxFault_model_1.2k``
     - ~1,174
     - ~8,233
     - 4-layer network, progressive channel reduction (12→12→8→8)
   * - ``GearboxFault_model_1.5k``
     - ~1,914
     - ~8,878
     - 3-layer network, constant channels (16→16→16)

Both models use INT8 quantization with 4-channel, 256-sample input.

**Expected Accuracy**: 97–100% depending on model selection.

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'MSPM0G5187'

   dataset:
     dataset_name: 'gearbox_fault_detection'

   training:
     model_name: 'GearboxFault_model_1.2k'
     training_epochs: 30
     batch_size: 32
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

   testing: {}
   compilation: {}

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/gearbox_fault_detection/config_MSPM0.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\gearbox_fault_detection\config_MSPM0.yaml

See Also
--------

* :doc:`motor_bearing_fault` — Bearing fault classification from vibration data
* :doc:`fan_blade_fault_classification` — Fan blade fault classification
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/gearbox_fault_detection>`_
* `SpectraQuest Dataset on Kaggle <https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis>`_
* `TI Neural Network Compiler User Guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
