=============
PIR Detection
=============

Edge AI solution for PIR sensor-based motion classification on the MSPM0G5187
with NPU acceleration.

Overview
--------

This example demonstrates an Edge AI solution for classifying Passive Infrared
(PIR) sensor signals into motion categories. By leveraging machine learning on
the MSPM0G5187 with its integrated NPU, the system can distinguish between
different types of motion, significantly reducing false positives caused by
pets and environmental factors.

**Application**: Security systems, smart home automation, occupancy sensing

**Task Type**: Time Series Classification

**Data Type**: Multivariate (PIR sensor signals)

Device Support
--------------

The primary target device is the **MSPM0G5187**. The following devices are also
fully supported:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Device
     - Description
     - Configuration File
   * - ``MSPM0G5187``
     - MSPM0 with NPU (primary)
     - ``config_MSPM0.yaml``
   * - ``CC2755``
     - Wireless MCU
     - ``config.yaml``
   * - ``CC1352``
     - Multi-protocol wireless MCU
     - ``config_CC1352.yaml``
   * - ``CC1354``
     - Sub-GHz + BLE MCU
     - ``config_CC1354.yaml``
   * - ``CC35X1``
     - Wi-Fi MCU
     - ``config_CC35X1.yaml``

Check the ``config_<device>.yaml`` files for device-specific configurations.

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ with integrated NPU
* EdgeAI Sensor Boosterpack (`TIDA-010997 <https://www.ti.com/tool/TIDA-010997>`_) with PIR sensor

**Software**

* Code Composer Studio (CCS) 12.x or later
* MSPM0 SDK 2.08.00 or later (`MSPM0-SDK <https://www.ti.com/tool/MSPM0-SDK>`_)
* TI Edge AI Studio

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         # Run with MSPM0 configuration (primary)
         ./run_tinyml_modelzoo.sh examples/pir_detection/config_MSPM0.yaml

         # Run with other device configurations
         ./run_tinyml_modelzoo.sh examples/pir_detection/config.yaml
         ./run_tinyml_modelzoo.sh examples/pir_detection/config_CC1352.yaml
         ./run_tinyml_modelzoo.sh examples/pir_detection/config_CC1354.yaml
         ./run_tinyml_modelzoo.sh examples/pir_detection/config_CC35X1.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         # Run with MSPM0 configuration (primary)
         .\run_tinyml_modelzoo.bat examples\pir_detection\config_MSPM0.yaml

         # Run with other device configurations
         .\run_tinyml_modelzoo.bat examples\pir_detection\config.yaml
         .\run_tinyml_modelzoo.bat examples\pir_detection\config_CC1352.yaml
         .\run_tinyml_modelzoo.bat examples\pir_detection\config_CC1354.yaml
         .\run_tinyml_modelzoo.bat examples\pir_detection\config_CC35X1.yaml

Dataset Description
-------------------

The ``pir_detection_classification`` dataset contains PIR sensor data captured
using the EdgeAI Sensor Boosterpack from a range of 6-8 meters.

**Download**: `pir_detection_classification_dsk.zip <https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/pir_detection_classification_dsk.zip>`_

**Classes** (3):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - Human Motion
     - Motion patterns produced by humans
   * - Background Motion
     - Environmental and non-living motion sources
   * - Dog Motion
     - Motion patterns produced by dogs

Feature Extraction Pipeline
---------------------------

The feature extraction pipeline processes raw PIR sensor data through the
following stages:

1. **ADC Sampling** -- Raw sensor data acquisition
2. **DC Offset Removal** -- Removes baseline drift from the signal
3. **Windowed Processing** -- Segments the signal into analysis windows
4. **Symmetric Mirroring** -- Prepares the signal for FFT computation
5. **FFT Computation** -- Transforms signal to frequency domain
6. **Magnitude Calculation** -- Computes spectral magnitudes
7. **Average Pooling** -- Reduces dimensionality of frequency features
8. **Additional Features** -- Extracts supplementary features:

   * Zero Crossing Rate (ZCR)
   * Slope Changes
   * Dominant Frequency

9. **Feature Concatenation** -- Combines all features into the final input vector

Model
-----

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Parameters
     - Description
   * - ``PIRDetection_model_1_t`` (Default)
     - ~53K+
     - Compact CNN, NPU compatible, optimized for multivariate input signals

Expected Results
----------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Configuration
     - Accuracy
   * - CC1352 with floating point feature extraction
     - ~98%
   * - MSPM0 with fixed point feature extraction
     - ~92.46%

Training Configuration
----------------------

The default training hyperparameters are:

.. code-block:: text

   batch_size:    64
   lr:            0.00001
   optimizer:     Adam
   weight_decay:  1e-20
   epochs:        100

**Quantization**: INT8 quantization is applied for NPU compatibility on the
MSPM0G5187.

**Compilation**: The TI Neural Network Compiler (`TI NNC <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_)
generates the compiled artifacts ``model.a`` and ``tvmgen_default.h`` for
on-device deployment.

References
----------

* `MSPM0G5187 Product Page <https://www.ti.com/product/MSPM0G5187>`_
* `TIDA-010997 EdgeAI Sensor Boosterpack <https://www.ti.com/tool/TIDA-010997>`_
* `TI Neural Network Compiler (NNC) User Guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main>`_
