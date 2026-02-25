=================
Device Deployment
=================

This section guides you through deploying trained models on TI microcontrollers
using Code Composer Studio (CCS).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   ccs_integration
   npu_device_deployment
   non_npu_deployment

Supported Devices
-----------------

**Device-Specific SDKs**

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Device Family
     - Devices
     - SDK
     - SDK Version
     - Download
   * - C2000 (F28x)
     - F28003x, F28004x, F28P55x, F28P65x
     - C2000Ware
     - 6.01.00.00
     - `C2000Ware <https://www.ti.com/tool/C2000WARE>`__
   * - C2000 (F29x)
     - F29H85x, F29P58x, F29P32x
     - F29H85X SDK
     - 01.04.00.00
     - `F29H85X-SDK <https://www.ti.com/tool/download/F29H85X-SDK/>`__
   * - MSPM33
     - MSPM33C321Ax
     - MSPM33 SDK
     - 1.03.00.00
     - `MSPM33-SDK <https://www.ti.com/tool/download/MSPM33-SDK>`__
   * - Sitara MCU (AM13x)
     - AM13E2x
     - MCU SDK
     - 1.00.00.00
     - `MCU-SDK-AM13E2X <https://www.ti.com/tool/download/MCU-SDK-AM13E2X>`__
   * - Sitara MCU (AM26x)
     - AM263, AM263P, AM261
     - MCU-PLUS-SDK
     - 11.03.00.00
     - `MCU-PLUS-SDK-AM263X <https://www.ti.com/tool/MCU-PLUS-SDK-AM263X>`__
   * - Connectivity
     - CC2755, CC1352, CC1354, CC35X1
     - SimpleLink SDK
     - --
     - `SimpleLink <https://www.ti.com/tool/SIMPLELINK-LOWPOWER-F3-SDK>`__

**Supported LaunchPads/EVMs**

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Device
     - LaunchPad/EVM
     - Product Page
     - NPU
   * - F28003x
     - `LAUNCHXL-F280039C <https://www.ti.com/tool/LAUNCHXL-F280039C>`__
     - `TMS320F280039C <https://www.ti.com/product/TMS320F280039C>`__
     - No
   * - F28004x
     - `LAUNCHXL-F280049C <https://www.ti.com/tool/LAUNCHXL-F280049C>`__
     - `TMS320F280049C <https://www.ti.com/product/TMS320F280049C>`__
     - No
   * - F28P55x
     - `LAUNCHXL-F28P55X <https://www.ti.com/tool/LAUNCHXL-F28P55X>`__
     - `TMS320F28P550SJ <https://www.ti.com/product/TMS320F28P550SJ>`__
     - Yes
   * - F28P65x
     - `LAUNCHXL-F28P65X <https://www.ti.com/tool/LAUNCHXL-F28P65X>`__
     - `TMS320F28P650DK <https://www.ti.com/product/TMS320F28P650DK>`__
     - No
   * - F29H85x
     - `F29H85X-SOM-EVM <https://www.ti.com/tool/F29H85X-SOM-EVM>`__
     - `F29H850TU <https://www.ti.com/product/F29H850TU>`__
     - No
   * - MSPM33C321Ax
     - `LP-MSPM33C321A <https://www.ti.com/tool/LP-MSPM33C321A>`__
     - `MSPM33C321A <https://www.ti.com/product/MSPM33C321A>`__
     - No
   * - AM13E2x
     - --
     - `AM13E2 <https://www.ti.com/product/AM13E2>`__
     - Yes
   * - AM263x
     - `LP-AM263 <https://www.ti.com/tool/LP-AM263>`__
     - `AM2634 <https://www.ti.com/product/AM2634>`__
     - No
   * - CC2755
     - --
     - `CC2755 <https://www.ti.com/product/CC2755>`__
     - No
   * - CC1352
     - --
     - `CC1352 <https://www.ti.com/product/CC1352R>`__
     - No
   * - CC1354
     - --
     - `CC1354 <https://www.ti.com/product/CC1354R7>`__
     - No
   * - CC35X1
     - --
     - `CC35X1 <https://www.ti.com/product/CC3511>`__
     - No

Deployment Overview
-------------------

After training and compilation, Tiny ML Tensorlab produces artifacts that you
can integrate into your CCS project:

**Output Artifacts**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``mod.a``
     - Compiled model library (ONNX model compiled by TI MCU NNC into C code,
       packaged as a static library)
   * - ``tvmgen_default.h``
     - Header file with model inference API exposed by ``mod.a``
   * - ``test_vector.c``
     - Golden test vectors (input + expected output) for on-device validation
   * - ``user_input_config.h``
     - Feature extraction configuration (preprocessing flags and parameters)

**Deployment Steps**

1. Train and compile your model using Tiny ML Tensorlab
2. Open Code Composer Studio
3. Import the SDK example project for your target device and task type
4. Copy the four output files from ModelMaker to the CCS project
5. Build, flash, and debug

The specific steps vary depending on whether your device has an NPU:

* :doc:`npu_device_deployment` - For F28P55x, AM13E2x, MSPM0G5187
* :doc:`non_npu_deployment` - For all other devices

Prerequisites
-------------

**Code Composer Studio (CCS)**

Download from https://www.ti.com/tool/CCSTUDIO. Version 20.2.0 or later
is recommended.

Install support for your target device family during CCS setup, and then
install the appropriate SDK listed in the `Supported Devices`_ table above.

Output File Locations
---------------------

After ModelMaker runs, the four deployment files are located in:

.. code-block:: text

   tinyml-modelmaker/data/projects/{dataset_name}/run/{date-time}/{model_name}/
   ├── compilation/
   │   └── artifacts/
   │       ├── mod.a                     <-- Compiled model
   │       └── tvmgen_default.h          <-- Model API header
   │
   └── training/
       ├── base/                         <-- For float (non-quantized) models
       │   └── golden_vectors/
       │       ├── test_vector.c
       │       └── user_input_config.h
       │
       └── quantization/                 <-- For quantized models
           └── golden_vectors/
               ├── test_vector.c
               └── user_input_config.h

.. important::

   Choose the golden vectors based on the model type you compiled:

   * **Float model (non-quantized):** Use golden vectors from
     ``training/base/golden_vectors/``
   * **Quantized model:** Use golden vectors from
     ``training/quantization/golden_vectors/``

   The ``mod.a`` and ``tvmgen_default.h`` are always from
   ``compilation/artifacts/`` regardless of quantization setting.

CCS Example Project Locations
------------------------------

TI provides ready-to-use CCS example projects for each task type. The example
project location depends on your device family:

**C2000Ware (F28x devices):**

.. code-block:: text

   {C2000WARE_INSTALL_PATH}/libraries/ai/examples/
   ├── generic_timeseries_classification/
   ├── generic_timeseries_regression/
   ├── generic_timeseries_forecasting/
   └── generic_timeseries_anomalydetection/

**F29H85X SDK (F29x devices):**

.. code-block:: text

   {F29H85X_SDK_INSTALL_PATH}/examples/rtlibs/ai/examples/
   ├── generic_timeseries_classification/
   ├── generic_timeseries_regression/
   ├── generic_timeseries_forecasting/
   └── generic_timeseries_anomalydetection/

Inside each task-type folder, select the subfolder matching your target device
(e.g., ``f28p55x/``, ``f28004x/``, ``f29h85x/``).

Task-Type-Specific Deployment Notes
------------------------------------

The deployment workflow is the same for all task types, but the CCS example
project names and verification differ:

**Classification**

* CCS example: ``generic_timeseries_classification``
* Verification: Check ``test_result`` variable (1 = pass, 0 = fail)
* The model outputs class scores; ``argmax`` gives the predicted class

**Regression**

* CCS example: ``generic_timeseries_regression``
* Verification: Check ``test_result`` variable (1 = pass, 0 = fail)
* The model outputs a continuous value

**Forecasting**

* CCS example: ``generic_timeseries_forecasting``
* Verification: Check ``test_result`` variable (1 = pass, 0 = fail)
* The model predicts future values from historical data

**Anomaly Detection**

* CCS example: ``generic_timeseries_anomalydetection``
* Verification: The application computes reconstruction error (MSE between
  input and autoencoder output) and compares to a threshold
* In ``user_input_config.h``, the anomaly threshold is defined:

  .. code-block:: c

     #define ANOMALY_THRESHOLD 0.014  // Threshold for k=4.5

  This was calculated during training as:
  ``threshold = mean_normal_error + k * std_normal_error``

* **Adjusting the threshold:**

  - Increase threshold (higher k): fewer false alarms, may miss subtle anomalies
  - Decrease threshold (lower k): catches more anomalies, more false alarms

  Refer to the ``threshold_performance.csv`` file from ModelMaker to choose the
  optimal k value for your application.

Testing Multiple Cases
----------------------

The ``test_vector.c`` file contains multiple test cases (SET 0, SET 1, etc.).
To test different cases:

1. Open ``test_vector.c`` in the CCS editor
2. Comment out the current test set (SET 0)
3. Uncomment another test set (e.g., SET 1)
4. Rebuild and reflash the project
5. Verify that ``test_result`` matches expected output

Model Compilation Details
-------------------------

The ONNX model produced by training is compiled using
`TI MCU NNC <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`__
to generate ``mod.a`` and ``tvmgen_default.h``. The compilation command
used internally is:

.. code-block:: bash

   tvmc compile \
     --target="c, ti-npu type=hard skip_normalize=true output_int=true" \
     --target-c-mcpu=c28 \
     ./model.onnx \
     -o artifacts_c28/mod.a \
     --cross-compiler="cl2000" \
     --cross-compiler-options="$CL2000_OPTIONS"

Different task types use different compilation flags:

* **Classification:** ``skip_normalize=true output_int=true``
* **Regression/Forecasting:** Output is float (no ``output_int``)
* **Anomaly Detection:** Output is reconstruction error

See the `TI MCU NNC documentation
<https://software-dl.ti.com/mctools/nnc/mcu/users_guide/compiling.html>`__
for all compilation options.
