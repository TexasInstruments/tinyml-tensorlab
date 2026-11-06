==============
NPU Guidelines
==============

Some TI MCUs include a hardware neural processing unit, the TI NPU (TinyEngine NPU).
This page summarizes which devices have it, how a model ends up running on it, and the
key layer rules. The complete layer-by-layer constraints are in
:doc:`/NPU_CONFIGURATION_GUIDELINES`.

NPU Devices
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - ``target_device``
     - Family
     - Notes
   * - **F28P55**
     - C2000 (C28x)
     - Time series tasks. NPU set up by SysConfig in the C2000Ware examples.
   * - **MSPM0G5187**
     - MSPM0 (Cortex-M0+)
     - Time series, image and audio tasks.
   * - **AM13E2**
     - AM13 (Cortex-M33)
     - Time series and image tasks.
   * - MSPM33C34
     - MSPM33C (Cortex-M33)
     - Has an NPU device profile, but no 1.5.0 model lists it (see
       :doc:`connectivity_devices`).

CC2755, CC2745 and CC35X1 do **not** have an NPU. They accelerate the model with the
Cortex-M33 Custom Datapath Extension (``ti-npu type=m33cde``), see
:doc:`connectivity_devices`.

How a Model Runs on the NPU
---------------------------

Three things must line up:

1. **An NPU device** in ``common.target_device``.
2. **A TINPU-quantized model.** On NPU devices ``training.auto_quantization`` is on by
   default and produces the quantized model under ``training/quantization/``. If the
   final model is not TINPU-quantized, ModelMaker compiles it with
   ``ti-npu type=soft`` (CPU) even on an NPU device. See :doc:`/features/quantization`.
3. **A hardware NPU preset**: ``default_preset`` or ``compress_npu_layer_data``.
   ``forced_soft_npu_preset`` deliberately runs the model on the CPU.

.. code-block:: yaml

   common:
     task_type: arc_fault
     target_device: F28P55
   training:
     model_name: CLS_1k_NPU
   compilation:
     compile_preset_name: default_preset   # or compress_npu_layer_data to save Flash

``compress_npu_layer_data`` adds ``opt_for_space=true`` to the compiler target, which
compresses the NPU layer data. All presets are described in
:doc:`device_overview`.

You can tell that a compiled model uses the NPU from ``tvmgen_default.h``: it defines
``TVMGEN_DEFAULT_TI_NPU`` (hardware NPU) or ``TVMGEN_DEFAULT_TI_NPU_SOFT`` (CPU). The SDK
examples check ``TVMGEN_DEFAULT_TI_NPU`` and, when it is defined, wait for
``tvmgen_default_finished`` after ``tvmgen_default_run()``. See
:doc:`/deployment/npu_device_deployment`.

Key Layer Rules
---------------

A layer is accelerated only if it meets the NPU constraints; otherwise it falls back
to software and the model gets slower. The most important rules:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Rule
     - Detail
   * - Channels in multiples of 4
     - Input and output channels of GCONV, DWCONV, PWCONV, MAXPOOL and AVGPOOL layers
       must be 4, 8, 12, 16, ...
   * - First convolution (FCONV)
     - Input channels = 1, output channels a multiple of 4, kernel width 1 to 8.
   * - Depthwise convolution
     - ``groups`` equal to the input channels.
   * - Pointwise convolution
     - Kernel ``(1, 1)``, stride ``(1, 1)``.
   * - Max pooling
     - At least one kernel dimension must be 1 to 4 (``(8, 1)`` is valid,
       ``(8, 8)`` is not).
   * - Global average pooling
     - Input must satisfy iH × iW > 2.
   * - Fully connected
     - At least 16 input features with 8-bit weights, at least 8 with 4-bit weights.

The ``_NPU`` models in the ModelZoo (``CLS_*_NPU``, ``REGR_*_NPU``, ``AD_*_NPU``,
``FCST_*_NPU``, the audio and MobileNet image models) are designed to meet these
rules, and keep all kernel sizes at 7 or below. The detailed guidelines describe the
cases where larger one-dimensional kernels are allowed.

.. seealso::

   * :doc:`/NPU_CONFIGURATION_GUIDELINES`: every supported layer type, design
     patterns, common pitfalls and a model design checklist.
   * :doc:`/BRING_YOUR_OWN_MODEL`: adds an NPU-friendly model to the ModelZoo step by step.
   * `TI MCU Neural Network Compiler user's guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_:
     the authoritative list of NPU layer configurations.

Troubleshooting
---------------

**The model runs, but slower than expected on an NPU device.**
Open the run's ``compilation/artifacts/tvmgen_default.h``. It defines
``TVMGEN_DEFAULT_TI_NPU`` when the model was compiled for the hardware NPU, and
``TVMGEN_DEFAULT_TI_NPU_SOFT`` when it was compiled for the CPU. In the second case,
make sure the model is TINPU-quantized and that you did not select
``forced_soft_npu_preset``.

**Some layers are not accelerated.**
Compare each layer with the rules above, typically a channel count that is not a
multiple of 4, a max-pooling kernel with both dimensions above 4, or fewer than 16
features into the FC layer.

**The application hangs after** ``tvmgen_default_run()``.
The NPU was not initialized. Keep the NPU initialization of the SDK example (see
:doc:`/deployment/npu_device_deployment`).
