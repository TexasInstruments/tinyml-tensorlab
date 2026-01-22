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
     - Compiled model library
   * - ``tvmgen_default.h``
     - Header file with model inference API
   * - ``test_vector.c``
     - Test vectors for device validation
   * - ``user_input_config.h``
     - Feature extraction configuration for device

**Deployment Steps**

1. Train and compile your model using Tiny ML Tensorlab
2. Open Code Composer Studio
3. Import or create a project for your target device
4. Copy the compiled artifacts to your project
5. Build, flash, and debug

The specific steps vary depending on whether your device has an NPU:

* :doc:`npu_device_deployment` - For F28P55, AM13E2, MSPM0G5187
* :doc:`non_npu_deployment` - For all other devices

Prerequisites
-------------

* Code Composer Studio installed (version 20.x or later recommended)
* Appropriate SDK installed:

  * **C2000 devices**: C2000Ware SDK
  * **MSPM0 devices**: MSPM0 SDK

* TI compiler tools configured (environment variables set)
