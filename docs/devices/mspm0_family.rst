=============
MSPM0 Family
=============

The MSPM0 family uses the Arm Cortex-M0+ core and targets low-cost, low-power designs:
battery-powered sensors, consumer products and always-on sensing. MSPM0G5187 adds the
TI NPU.

Supported Devices
-----------------

.. list-table::
   :header-rows: 1
   :widths: 16 10 74

   * - Device
     - NPU
     - Supported tasks
   * - **MSPM0G5187**
     - **Yes**
     - Time series classification (generic, arc fault, motor fault, ECG, PIR),
       regression, forecasting, anomaly detection, **image classification**,
       **audio classification**
   * - MSPM0G3507
     - No
     - Time series classification (generic, arc fault, motor fault, ECG, PIR),
       regression, forecasting, anomaly detection, image classification
   * - MSPM0G3519
     - No
     - Same as MSPM0G3507

The Cortex-M0+ has no floating-point unit; ModelMaker compiles with
``-mfloat-abi=soft``. For this reason most MSPM0 example configurations use the
fixed-point (Q15) feature-extraction presets, for example
``FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth`` or
``Custom_Default_MSPM0``. See :doc:`/features/feature_extraction`.

Feature-extraction memory is estimated for MSPM0G5187 and MSPM0G3507 (see
:doc:`device_overview`).

MSPM0G5187 (NPU)
----------------

MSPM0G5187 is the most-used target in the 1.5.0 examples: most examples ship a
``config_MSPM0.yaml`` for it. For example,
`examples/generic_timeseries_classification/config_MSPM0.yaml <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/generic_timeseries_classification>`_:

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: hello_world_example_dsg
   data_processing_feature_extraction:
     feature_extraction_name: FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth
   training:
     model_name: CLS_13k_NPU

Use ``compile_preset_name: compress_npu_layer_data`` to reduce Flash use, or
``forced_soft_npu_preset`` to run on the CPU instead of the NPU. See
:doc:`npu_guidelines`.

**Audio classification** (MSPM0G5187 only) uses the ``DSCNN_NPU``, ``DSCNN_32K_NPU``,
``DSCNN_GB_NPU``, ``TCDS_ResNet_NPU`` and ``TCDS_ResNet_FB_NPU`` models, for example in
`wake_word_detection <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/wake_word_detection>`_,
`glass_break_detection <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/glass_break_detection>`_,
`cough_detection <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/cough_detection>`_ and
`google_speech_command <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/google_speech_command>`_.

**Image classification** on MSPM0 uses ``Lenet5``, ``MobileNetV1_28k_NPU`` and
``MobileNetV2_58k_NPU``, for example in
`MNIST_image_classification <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/MNIST_image_classification>`_
and `coffee_bean_classification <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/coffee_bean_classification>`_.
See :doc:`/task_types/image_classification`.

MSPM0G3507 and MSPM0G3519
-------------------------

These devices run the model on the Cortex-M0+ CPU. Use the same models and
configurations as for MSPM0G5187 and change ``target_device``:

.. code-block:: yaml

   common:
     target_device: MSPM0G3507

Keep models small: without an NPU the inference time grows with the model's operation
count. Check memory before training with ``./estimate_memory.sh <config.yaml>``.

A few models are MSPM0-only: ``ElectricalFault_model_40k_t``,
``GearboxFault_model_1.2k_t`` and ``GearboxFault_model_1.5k_t``.

Running on the Board
--------------------

.. important::

   MSPM0 devices are **not** covered by the SDK examples described in
   :doc:`/DEPLOYING_TO_SDK_EXAMPLE` (C2000Ware, F29H85x SDK, MSPM33 SDK, MCU-SDK and
   MCU-PLUS-SDK). The guidance below is the MSPM0 guidance carried over from 1.4.0
   and the links given in the 1.5.0 example readmes.

* Install `Code Composer Studio <https://www.ti.com/tool/CCSTUDIO>`_ with MSPM0 support
  and the `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_. (The
  `pir_detection <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/pir_detection>`_
  example asks for MSPM0 SDK 2.08.00 or later.)
* Follow the
  `EdgeAI Software Guide in the MSPM0 SDK <https://dev.ti.com/tirex/explore/node?node=A__AKCnvqDed-Plz2JO5Umb3Q__MSPM0-SDK__a3PaaoK__LATEST>`_
  for the MSPM0 AI example projects and how to integrate the ModelMaker output.
* The ModelMaker output is the same as for other devices: ``mod.a`` and
  ``tvmgen_default.h`` in ``compilation/artifacts/``, and ``test_vector.c`` and
  ``user_input_config.h`` in ``training/quantization/golden_vectors/``. See
  :doc:`/deployment/non_npu_deployment`.

Next Steps
----------

* :doc:`npu_guidelines` for MSPM0G5187 model constraints
* :doc:`/deployment/index` for running the model on a device
