==================
Non-NPU Deployment
==================

Devices without a hardware NPU run the compiled model on the CPU. The deployment procedure is the
same as for NPU devices: four files from one run go into an SDK example
(:doc:`/DEPLOYING_TO_SDK_EXAMPLE`, :doc:`ccs_integration`). This page lists the differences.

Non-NPU Devices
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - ``target_device``
     - SDK example folder
     - Notes
   * - F28004, F28003
     - C2000Ware ``libraries/ai/examples/<example>/f28004x/`` or ``f28003x/``
     - C28x CPU
   * - F28P65
     - C2000Ware ``.../f28p65x/``
     - C28x CPU
   * - F29H85
     - F29H85x SDK ``examples/ai/<example>/f29h85x/``
     - C29x CPU; examples call ``TI_NPU_init()``
   * - MSPM33C32
     - MSPM33 SDK ``examples/ai/<example>/mspm33c321a/``
     - Cortex-M33
   * - AM263
     - MCU-PLUS-SDK ``examples/ai/<example>/am263x_lp/r5fss0-0_nortos/``
     - Cortex-R5F
   * - MSPM0G3507, CC devices, xWRL6432
     - Not in ``tinyml-sdk``
     - 1.4.0 guidance: integrate ``mod.a``, ``tvmgen_default.h`` and the feature-extraction
       library into your own project. See :doc:`/devices/mspm0_family` and
       :doc:`/devices/connectivity_devices`.

The complete device list is in :doc:`/devices/device_overview`.

Importing an Example That Is Not in Resource Explorer
-----------------------------------------------------

Some examples are not listed in CCS Resource Explorer. Import them manually:

1. **File -> Import Projects**.

.. figure:: /_static/img/deployment/non_npu/import_project_menu.png
   :width: 600px
   :align: center
   :alt: File Import Projects menu

   Selecting Import Projects from the File menu

2. **Browse** to the example's device folder and click **Finish**.

.. figure:: /_static/img/deployment/non_npu/browse_project.png
   :width: 600px
   :align: center
   :alt: Browse to project folder

   Browsing to the project folder

Then replace the four files, build, flash and check, exactly as in :doc:`ccs_integration`.

CPU Inference Considerations
----------------------------

* **Model size.** Without an NPU, cycles grow with the multiply-accumulate count of the model.
  Use models that list your device and keep them small.
* **Feature extraction runs on the CPU too.** The SDK feature-extraction library (FFT, binning,
  windowing) is configured entirely by ``user_input_config.h``. ModelMaker reports its code and
  data memory for many devices (:doc:`/devices/device_overview`).
* **Memory.** If the linker reports a section that does not fit, start from an example with
  a similar feature-extraction preset and adjust the ``lnk.cmd`` placement.
* **Timing.** Measure on target with a CPU timer around ``tvmgen_default_run`` and
  ``FE_runFeatureExtract`` in ``application_main.c``. The cycle and energy numbers of the 1.4.0
  guide are not carried over because they were not re-measured for 1.5.0.
* **Integer types.** If ``SKIP_NORMALIZE`` or ``OUTPUT_INT`` is defined in
  ``user_input_config.h``, ``model_input_t`` and ``model_output_t`` are ``int8_t``. Keep that
  file matched to ``mod.a``.

Non-NPU vs NPU
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - NPU devices
     - Non-NPU devices
   * - Inference
     - Hardware accelerated, waits on ``tvmgen_default_finished``
     - CPU only
   * - Initialization
     - NPU module in SysConfig (or ``TI_NPU_init()``)
     - None needed for the model
   * - Compile preset
     - ``default_preset`` / ``compress_npu_layer_data``
     - ``default_preset``
   * - Models
     - NPU-compatible (``*_NPU``)
     - Any model that lists the device

Next Steps
----------

* :doc:`npu_device_deployment`
* :doc:`ccs_integration`
* :doc:`/devices/device_overview`
