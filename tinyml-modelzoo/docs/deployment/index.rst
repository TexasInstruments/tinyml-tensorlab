==========
Deployment
==========

A Tiny ML ModelZoo run produces a compiled model library (``mod.a``), the model header
(``tvmgen_default.h``), the feature-extraction configuration (``user_input_config.h``) and
golden vectors (``test_vector.c``). Deploying means dropping these four files into a
ready-made example from the device SDK and confirming on the board that the model gives the same
answer as on your PC.

.. note::

   Deployment in 1.5.0 is based on the SDK examples in ``tinyml-sdk`` (C2000Ware,
   F29H85x SDK, MSPM33 SDK, AM13 MCU-SDK, AM26x MCU-PLUS-SDK). The 1.4.0 flow of writing your
   own ``main.c`` around ``mod_inference()`` and ``extract_features()`` is gone; the SDK
   examples already contain the application, the feature-extraction library calls and the
   test harness.

Start Here
----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Deploy to an SDK Example
      :link: /DEPLOYING_TO_SDK_EXAMPLE
      :link-type: doc

      The 4 files, which SDK and example to pick, and the expected "All tests have passed." output.

   .. grid-item-card:: CCS Integration
      :link: ccs_integration
      :link-type: doc

      Importing, building, flashing and debugging the example in Code Composer Studio.

   .. grid-item-card:: NPU Devices
      :link: npu_device_deployment
      :link-type: doc

      What changes for F28P55, MSPM0G5187 and AM13E2 (NPU initialization, memory, debugging).

   .. grid-item-card:: Non-NPU Devices
      :link: non_npu_deployment
      :link-type: doc

      CPU-only devices: C28x, F29H85, MSPM0G3507, MSPM33C32, AM26x.

The deck `Deploying to an SDK example <../slides/deploying_to_sdk_example.html>`_ summarizes
the same steps as slides.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   /DEPLOYING_TO_SDK_EXAMPLE
   ccs_integration
   npu_device_deployment
   non_npu_deployment
   /deploying_classification_models_from_modelmaker_to_device/readme
   /deploying_regression_models_from_modelmaker_to_device/readme
   /deploying_forecasting_models_from_modelmaker_to_device/readme
   /deploying_anomaly_detection_models_from_modelzoo_to_device/readme

Deployment Workflow
-------------------

1. Train and compile with a config that has ``compilation: enable: True`` (see
   :doc:`/getting_started/quickstart`). Check the device-support matrix in :doc:`/devices/device_overview`.
2. Open the run folder ``data/projects/<project_name>/run/<date-time>/<model_name>/`` (see
   :doc:`/getting_started/understanding_run_output`).
3. Pick the SDK and example for your ``target_device`` and ``task_type``
   (tables in :doc:`/DEPLOYING_TO_SDK_EXAMPLE`).
4. Import the example in CCS, replace the four files, build, flash.
5. Read the console: ``Feature extraction mismatches 0``, ``All tests have passed.``,
   ``Golden vectors matched: X not matched: 0``.
6. Replace the test vector input with your ADC or sensor data in ``application_main.c``.

Per-task walkthroughs (with screenshots) are the four task readmes in the toctree above.
