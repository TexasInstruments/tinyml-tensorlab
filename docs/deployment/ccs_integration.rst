=====================
CCS Integration Guide
=====================

This page covers the Code Composer Studio (CCS) side of deployment: importing an SDK example,
building, flashing and debugging it. What to copy from the run folder, and which SDK example
to pick, is described once in :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

Prerequisites
-------------

* **Code Composer Studio** 20.4.0 or later (the version used by the 1.5.0 deploy guide), with
  support for your device family.
* The **device SDK** that holds the AI examples (see the SDK table in
  :doc:`/DEPLOYING_TO_SDK_EXAMPLE`), installed through CCS Resource Explorer or standalone.
* A **trained and compiled run** (``compilation: enable: True``). The files you need:

  .. code-block:: text

     compilation/artifacts/mod.a
     compilation/artifacts/tvmgen_default.h
     training/quantization/golden_vectors/test_vector.c
     training/quantization/golden_vectors/user_input_config.h

  ``compilation/<date-time>_<model>_<device>.zip`` bundles the same set.

What You Get in the Run Folder
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Purpose
   * - ``mod.a``
     - Compiled model library for one device (instruction set and NPU or not). On AM13E2 the project
       links ``artifacts/mod_m33_ti_arm_clang.a``, so rename the copy.
   * - ``tvmgen_default.h``
     - Model API (``tvmgen_default_run``) plus input normalization parameters.
   * - ``user_input_config.h``
     - ``#define`` values that configure the SDK feature-extraction library to match training.
       ``SKIP_NORMALIZE`` and ``OUTPUT_INT`` switch ``model_input_t`` / ``model_output_t`` to ``int8_t``.
   * - ``test_vector.c``
     - Raw input, expected model input and expected model output for a few samples.

Import the Example
------------------

1. Open CCS and go to **View -> Resource Explorer**.

.. figure:: /_static/img/deployment/npu/resource_explorer.png
   :width: 700px
   :align: center
   :alt: CCS Resource Explorer

   Resource Explorer in Code Composer Studio

2. Filter by your board (for example ``LAUNCHXL-F28P55X``) and search for the example name.

.. figure:: /_static/img/deployment/npu/resource_explorer_with_filled_fields.png
   :width: 700px
   :align: center
   :alt: Resource Explorer with filters

   Board and keyword filters

3. Select the example and click **Import**; install any dependencies when prompted.

.. figure:: /_static/img/deployment/npu/import_project.png
   :width: 500px
   :align: center
   :alt: Import project

   Importing the example

.. figure:: /_static/img/deployment/npu/download_dependencies.png
   :width: 500px
   :align: center
   :alt: Download dependencies

   Installing required dependencies

The project spec copies ``artifacts/``, ``test_vector.c``, ``user_input_config.h`` and
``application_main.c`` into your CCS workspace. Edit the workspace copy; the SDK install stays untouched.
If an example is not listed in Resource Explorer, use **File -> Import Projects** and browse to the example's
device folder (see :doc:`non_npu_deployment`).

Replace the four files in the workspace project with the ones from your run. Do not change
``application_main.c``, the ``.syscfg`` or the linker command file unless you need to (see
:ref:`ccs-memory`).

Build, Flash and Run
--------------------

1. **Project -> Build Project(s)** (``Ctrl+B``) and check the Console for errors.

.. figure:: /_static/img/deployment/npu/build_project.png
   :width: 600px
   :align: center
   :alt: Build project

   Building the project

2. Set the LaunchPad ``.ccxml`` (for example ``TMS320F28P550SJ9_LaunchPad.ccxml``) as the active
   target configuration: right-click it in Project Explorer.

.. figure:: /_static/img/deployment/npu/active_target.png
   :width: 500px
   :align: center
   :alt: Active target configuration

   Selecting the LaunchPad target configuration

3. Connect the board over USB and use **Run -> Debug Project** (or **Flash Project**). If a
   firmware-update dialog appears, click **Update**.

.. figure:: /_static/img/deployment/npu/flash_application.png
   :width: 600px
   :align: center
   :alt: Flash application

   Flashing the application

.. figure:: /_static/img/deployment/npu/update_error.png
   :width: 500px
   :align: center
   :alt: Firmware update dialog

   Firmware update dialog

Check the Result
----------------

1. In the Debug perspective, add ``test_result`` to the **Watch** window and press **Resume** (F8).

.. figure:: /_static/img/deployment/npu/debug_screen.png
   :width: 700px
   :align: center
   :alt: Debug screen

   CCS Debug perspective

.. figure:: /_static/img/deployment/npu/variable_test_result.png
   :width: 600px
   :align: center
   :alt: Watch test_result

   Adding ``test_result`` to the Watch window

2. Read the CIO console. For classification, regression and forecasting:

   .. code-block:: text

      Feature extraction mismatches 0
      All tests have passed.
      Golden vectors matched: 2 not matched: 0

   ``test_result`` is 1 on PASS. Values may differ by at most 1 from the PC result (feature
   extraction and model output). The "matched" count equals the number of model outputs.

.. figure:: /_static/img/deployment/npu/test_result_value.png
   :width: 600px
   :align: center
   :alt: test_result value

   ``test_result`` after a passing run

For anomaly detection the tolerance on each output is 4, the example then computes the
reconstruction error and compares it with ``RECONSTRUCTION_ERROR_THRESHOLD`` from
``user_input_config.h``. There ``test_result`` means 1 = anomaly, 0 = normal.

To step through the inference yourself, place a breakpoint after the ``tvmgen_default_run`` call in
``application_main.c`` and inspect ``model_output``.

.. figure:: /_static/img/deployment/npu/breakpoint.png
   :width: 700px
   :align: center
   :alt: Breakpoint

   Breakpoint after the inference call

.. _ccs-memory:

Memory and Linker Problems
--------------------------

If the linker reports that a section does not fit (for example ``FFT_buffer_1`` or ``.bss``),
your model or feature-extraction configuration is larger than the example's. Start from an
example with a similar feature-extraction preset, or adjust the memory placement in
``lnk.cmd``. ModelMaker reports model memory (and, for supported devices, feature-extraction library
memory) after compilation; see :doc:`/devices/device_overview`.

Common Issues
-------------

The full symptom table (FE mismatches, ``not matched`` counts, ``model_input_t`` build errors,
undefined ``tvmgen_default_run``, hang at ``tvmgen_default_finished``) is in the
troubleshooting section of :doc:`/DEPLOYING_TO_SDK_EXAMPLE`. In short: all four files must
come from the same run, golden vectors from ``training/quantization/``, and a stale build needs
**Project -> Clean**.

Next Steps
----------

* :doc:`npu_device_deployment` for NPU devices
* :doc:`non_npu_deployment` for CPU-only devices
* :doc:`/troubleshooting/common_errors`
