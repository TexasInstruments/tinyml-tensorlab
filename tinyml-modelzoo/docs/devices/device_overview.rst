===============
Device Overview
===============

This page lists every ``common.target_device`` value that Tiny ML ModelZoo |release|
accepts, with the facts you need to choose one: CPU core, hardware NPU, supported
tasks, whether a ready-made SDK example exists, and whether feature-extraction memory
is estimated.

.. note::

   The tables on this page are generated from the 1.5.0 source: the
   ``TARGET_DEVICE_*`` constants and device compilation profiles in ModelMaker
   (``tinyml_modelmaker/ai_modules/{timeseries,vision,audio,radar}/constants.py``)
   and the ``target_devices`` of each model in ``tinyml_modelzoo/model_descriptions``.
   For clock speed, memory size and peripherals, see the device's product page on
   `ti.com <https://www.ti.com/microcontrollers-mcus-processors/overview.html>`_.

Selecting the Device
--------------------

Set the device in the ``common`` section of the config YAML. Use the exact string from
the tables below (it is case sensitive):

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55

ModelMaker checks the combination before it starts. If the device is unknown, or no
model for the ``task_type`` supports the device, the run stops with an error that lists
the supported devices or tasks.

Complete Device List
--------------------

**Task legend:**
``C`` generic time series classification (``generic_timeseries_classification``),
``Arc`` arc fault, ``Motor`` motor fault, ``Blower`` blower imbalance,
``ECG`` ECG classification, ``PIR`` PIR detection,
``R`` regression, ``F`` forecasting, ``AD`` anomaly detection,
``Img`` image classification, ``Aud`` audio classification, ``Rad`` radar classification.

**SDK example** names the device folder of the ready-made examples in the TI SDKs
(see :doc:`/deployment/index`). **FE memory** means ModelMaker compiles the SDK
feature-extraction library for this device and reports its memory.

C2000 F28x (C28x DSP)
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 14 20 8 30 16 12

   * - Device
     - Core (compiler flags)
     - NPU
     - Tasks
     - SDK example
     - FE memory
   * - **F28P55**
     - C28x, FPU32
     - **Yes**
     - C, Arc, Motor, Blower, R, F, AD
     - C2000Ware ``f28p55x``
     - Yes
   * - F28P65
     - C28x, FPU64
     - No
     - C, Arc, Motor, Blower, R, F, AD
     - C2000Ware ``f28p65x``
     - Yes
   * - F28P551x
     - C28x, FPU32
     - No
     - C, Arc, Motor, Blower, ECG, R, F, AD
     - --
     - No
   * - F2837
     - C28x, FPU32
     - No
     - C, Arc, Motor, Blower, ECG, R, F, AD
     - --
     - Yes
   * - F2837xS
     - C28x, FPU32
     - No
     - C, Arc, Motor, Blower, ECG
     - --
     - No
   * - F2838x
     - C28x, FPU32
     - No
     - C, Arc, Motor, Blower, ECG, R, F, AD
     - --
     - No
   * - F2807x
     - C28x, FPU32
     - No
     - C, Arc, Motor, Blower, ECG
     - --
     - No
   * - F28003
     - C28x, FPU32
     - No
     - C, Arc, Motor, Blower, ECG, R, F, AD
     - C2000Ware ``f28003x``
     - Yes
   * - F28004
     - C28x, FPU32
     - No
     - C, Arc, Motor, Blower, ECG, R, F, AD
     - C2000Ware ``f28004x``
     - Yes
   * - F28002x
     - C28x, FPU32
     - No
     - Motor, Blower, ECG, R, F, AD
     - --
     - No
   * - F280013
     - C28x, FPU32
     - No
     - Arc, Motor, Blower, ECG, R, F, AD
     - --
     - Yes
   * - F280015
     - C28x, FPU32
     - No
     - Arc, Motor, Blower, ECG, R, F, AD
     - --
     - Yes
   * - F28E12
     - C28x, FPU32
     - No
     - R, F, AD
     - --
     - Yes

C2000 F29x (C29x DSP)
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 14 20 8 30 16 12

   * - Device
     - Core (compiler flags)
     - NPU
     - Tasks
     - SDK example
     - FE memory
   * - F29H85
     - C29x
     - No
     - C, Arc, Motor, Blower, R, F, AD
     - F29H85x SDK ``f29h85x``
     - Yes
   * - F29P58
     - C29x
     - No
     - C, R, F, AD
     - --
     - No
   * - F29P32
     - C29x
     - No
     - C, R, F, AD
     - --
     - No

Arm Cortex-M0+, Cortex-M33 and Cortex-R5F MCUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 14 20 8 30 16 12

   * - Device
     - Core (compiler flags)
     - NPU
     - Tasks
     - SDK example
     - FE memory
   * - MSPM0G3507
     - Cortex-M0+, no FPU
     - No
     - C, Arc, Motor, ECG, PIR, R, F, AD, Img
     - MSPM0 SDK (not in this flow, see :doc:`mspm0_family`)
     - Yes
   * - MSPM0G3519
     - Cortex-M0+, no FPU
     - No
     - C, Arc, Motor, ECG, PIR, R, F, AD, Img
     - MSPM0 SDK (see :doc:`mspm0_family`)
     - No
   * - **MSPM0G5187**
     - Cortex-M0+, no FPU
     - **Yes**
     - C, Arc, Motor, ECG, PIR, R, F, AD, Img, Aud
     - MSPM0 SDK (see :doc:`mspm0_family`)
     - Yes
   * - MSPM33C32
     - Cortex-M33, FPv5-SP
     - No
     - C, Arc, Motor, Blower, PIR, R, F, AD
     - MSPM33 SDK ``mspm33c321a``
     - No
   * - MSPM33C34
     - Cortex-M33, FPv5-SP
     - Yes (profile only)
     - None in 1.5.0 (no model lists this device)
     - --
     - No
   * - **AM13E2**
     - Cortex-M33, FPv5-SP
     - **Yes**
     - C, Arc, Motor, Blower, R, F, AD, Img
     - MCU-SDK (AM13E2x)
     - Yes
   * - AM263
     - Cortex-R5F, VFPv3-D16
     - No
     - C, Arc, Motor, Blower, R, F, AD
     - MCU-PLUS-SDK ``am263x_lp``
     - No
   * - AM263P
     - Cortex-R5F, VFPv3-D16
     - No
     - C, R, F, AD
     - --
     - No
   * - AM261
     - Cortex-R5F, VFPv3-D16
     - No
     - C, R, F, AD
     - --
     - No

Connectivity and Radar Devices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 14 20 8 30 16 12

   * - Device
     - Core (compiler flags)
     - NPU
     - Tasks
     - SDK example
     - FE memory
   * - CC2755
     - Cortex-M33 with CDE
     - No
     - C, Motor, PIR
     - --
     - No
   * - CC2745
     - Cortex-M33 with CDE
     - No
     - C, Motor, PIR
     - --
     - No
   * - CC35X1
     - Cortex-M33 with CDE
     - No
     - C, Motor, PIR
     - --
     - No
   * - CC1352
     - Cortex-M4F
     - No
     - C, Motor, PIR
     - --
     - No
   * - CC1312
     - Cortex-M4F
     - No
     - C, Motor, PIR
     - --
     - No
   * - CC1354
     - Cortex-M33, FPv5-SP
     - No
     - C, Motor, PIR
     - --
     - No
   * - CC1314
     - Cortex-M33, FPv5-SP
     - No
     - C, Motor, PIR
     - --
     - No
   * - xWRL6432
     - Cortex-M4F
     - No
     - Rad
     - --
     - No

.. note::

   * **MSPM33C34** has an NPU compilation profile and is accepted on the command line
     (it is hidden from the Model Composer GUI), but no 1.5.0 model lists it, so
     every task is rejected for it. Use MSPM33C32 for MSPM33 projects.
   * **CDE devices** (CC2755, CC2745, CC35X1) have no NPU. Their default preset
     compiles with ``ti-npu type=m33cde``, which uses the Cortex-M33 Custom Datapath
     Extension. See :doc:`connectivity_devices`.
   * Device-specific models exist too, for example ``WiFiPresDet_4.7k_CDE`` (CC35X1
     only) and the audio models (MSPM0G5187 only).

Compilation Presets
-------------------

ModelMaker compiles the trained ONNX model with the
`TI MCU Neural Network Compiler <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_.
The compile options come from a **preset**, selected with
``compilation.compile_preset_name``:

.. code-block:: yaml

   compilation:
     enable: True
     compile_preset_name: default_preset   # or compress_npu_layer_data, forced_soft_npu_preset

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Preset
     - Available on
     - What it does
   * - ``default_preset``
     - All devices
     - Recommended. Uses the hardware NPU when the device has one; otherwise compiles
       for the CPU. Options are chosen per task (for example, regression and
       forecasting do not request integer output).
   * - ``compress_npu_layer_data``
     - F28P55, MSPM0G5187, AM13E2 (and MSPM33C34, see note above)
     - Hardware NPU, plus ``opt_for_space=true``: compresses the NPU layer data to
       reduce Flash use.
   * - ``forced_soft_npu_preset``
     - F28P55, MSPM0G5187, AM13E2, CC2755, CC2745, CC35X1
     - Disables the hardware acceleration and compiles the model for the CPU
       (``ti-npu type=soft``).

If you ask for a preset that the device does not have, ModelMaker logs a warning and
uses ``default_preset``.

.. important::

   The hardware NPU only runs TINPU-quantized models. If the model is not
   TINPU-quantized (``training.quantization`` is not ``2``), ModelMaker replaces
   ``ti-npu type=hard`` with ``ti-npu type=soft`` and the model runs on the CPU, even on
   an NPU device. ``training.auto_quantization`` is on by default and is switched off
   automatically on devices without a hardware NPU (unless you set it in the config).
   See :doc:`/features/quantization`.

NPU and Non-NPU Devices
-----------------------

**Hardware NPU devices: F28P55, MSPM0G5187, AM13E2.**
The TI NPU accelerates convolution, pooling and fully connected layers of quantized
models. To be accelerated, a layer must meet the NPU constraints (channel multiples of 4,
kernel and pooling limits, minimum FC input size). Layers that do not meet them fall
back to software. See :doc:`npu_guidelines`.

**All other devices** run the model on the CPU. The ``_NPU`` models in the ModelZoo
(for example ``CLS_1k_NPU``) are not limited to NPU devices: they list CPU-only devices
in their ``target_devices`` as well, and ModelMaker compiles them for the CPU there.
Tiny ML ModelZoo 1.5.0 has no separate "non-NPU" variant of these models.

Feature-Extraction Memory Estimation
------------------------------------

New in 1.5.0: for the devices marked **FE memory: Yes** above, the compilation step
also compiles the SDK feature-extraction library with the run's
``user_input_config.h`` and reports its memory in the log:

.. code-block:: text

   ============================================================
   TI Feature Extraction Library Memory Usage
   ============================================================
   Code:                      ...
   RO Data:                   ...
   RW Data:                   ...

The devices are AM13E2, MSPM0G3507, MSPM0G5187, F29H85, F28P55, F28P65, F2837,
F28003, F28004, F280013, F280015 and F28E12. You can also estimate model and
feature-extraction memory before training with the standalone script in the
``tinyml-modelzoo`` folder:

.. code-block:: bash

   ./estimate_memory.sh examples/dc_arc_fault/config_dsk.yaml --target_device F28P55

See :doc:`/features/memory_estimation` for details.

Choosing a Device
-----------------

* **Start from the task.** Check the *Tasks* column. Image classification runs on
  MSPM0 and AM13E2, audio classification on MSPM0G5187, radar classification on
  xWRL6432, and F28E12 supports regression, forecasting and anomaly detection only.
* **Prefer an NPU device** (F28P55, MSPM0G5187, AM13E2) for larger convolutional models
  or tight latency budgets.
* **Check the SDK example column** if you want the quickest path to the board: those
  devices have ready-made SDK examples for all four time series tasks (see
  :doc:`/deployment/index`).
* **Check memory early** with ``estimate_memory.sh``; the device's Flash and RAM are on
  its ti.com product page.
* **Existing designs:** C2000 for motor control and power conversion, AM26x for
  industrial communication, CC devices for wireless sensors, MSPM0 for low-cost and
  low-power designs.
