=====================
NPU Device Deployment
=====================

This page lists what is specific to deploying on devices with a hardware NPU. The common
procedure (the four files, picking the SDK example, the expected console output) is in
:doc:`/DEPLOYING_TO_SDK_EXAMPLE`; the CCS steps are in :doc:`ccs_integration`.

NPU-Enabled Devices
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``target_device``
     - SDK example location
     - NPU initialization
   * - F28P55
     - C2000Ware ``libraries/ai/examples/<example>/f28p55x/``
     - SysConfig (NPU module in ``c2000.syscfg``)
   * - MSPM0G5187
     - MSPM0 SDK (not part of ``tinyml-sdk``; see :doc:`/devices/mspm0_family`)
     - See the MSPM0 SDK example
   * - AM13E2
     - MCU-SDK ``examples/ai/<example>/``
     - See the MCU-SDK example

MSPM33C34 has an NPU profile but no model lists it in 1.5.0
(:doc:`/devices/device_overview`). F29 devices (F29H85, F29P58, F29P32) have no hardware NPU: the
model is compiled with the *soft* NPU preset and runs on the CPU through TI's NPU software library.
That is why the F29H85 examples still call ``TI_NPU_init()`` in ``application_main.c`` and carry an
NPU module in their SysConfig; do not remove either from an example you import.

NPU Compilation
---------------

An NPU device is compiled with one of the NPU presets (``default_preset``,
``compress_npu_layer_data``); ``forced_soft_npu_preset`` deliberately runs the model on the CPU.
Model layers must satisfy the NPU rules in :doc:`/devices/npu_guidelines` and
:doc:`/NPU_CONFIGURATION_GUIDELINES`; models named ``*_NPU`` in the model list are designed for
that. The result is the same four files as for any device.

.. code-block:: yaml

   common:
     target_device: 'F28P55'

   compilation:
     enable: True
     compile_preset_name: default_preset   # or compress_npu_layer_data to save Flash

What Is Different at Run Time
-----------------------------

* **Initialization.** The NPU must be initialized before the first ``tvmgen_default_run``. In
  the SDK examples this is already done (SysConfig, or ``TI_NPU_init()`` for F29H85). Keep the
  NPU module in the project's ``.syscfg``.
* **Asynchronous completion.** On NPU devices the application waits for ``tvmgen_default_finished``
  before it compares the output. If the program hangs there, the NPU was not initialized.
* **Tolerance.** The golden-vector check uses the same tolerance (+/-1, or +/-4 for anomaly
  detection) as on CPU devices.
* **Memory.** Check the linker map after the build. ModelMaker also reports model and
  feature-extraction memory (see :doc:`/devices/device_overview`).

Walkthrough: F28P55x
--------------------

#. Install C2000Ware and CCS (:doc:`ccs_integration`).
#. Import ``f28p55x_arc_fault`` (or the example for your task) from Resource Explorer using the
   board filter ``LAUNCHXL-F28P55X``.
#. Replace ``artifacts/mod.a``, ``artifacts/tvmgen_default.h``, ``test_vector.c`` and
   ``user_input_config.h`` with the files from one NPU-compiled run (for example
   ``examples/dc_arc_fault/config_dsk.yaml`` with model ``CLS_1k_NPU``).
#. Build, select ``TMS320F28P550SJ9_LaunchPad.ccxml``, debug, add ``test_result`` to Watch and resume.
#. Expect ``Feature extraction mismatches 0`` and ``All tests have passed.``.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symptom
     - Fix
   * - Hangs at ``while (!tvmgen_default_finished)``
     - NPU not initialized. Keep the NPU module in ``.syscfg`` (C2000) or ``TI_NPU_init()`` (F29H85).
   * - Linker: incompatible object / undefined ``tvmgen_default_run``
     - ``mod.a`` was built for a different device. Re-run with the right ``target_device``.
   * - Model compiled but not on the NPU
     - A layer is not NPU-compatible, or ``forced_soft_npu_preset`` was used. See :doc:`/devices/npu_guidelines`.
   * - Section does not fit in Flash
     - Try ``compress_npu_layer_data`` or a smaller model.

Next Steps
----------

* :doc:`non_npu_deployment`
* :doc:`/devices/npu_guidelines`
