============
C2000 Family
============

C2000 real-time MCUs are TI's platform for motor control, power conversion and other
real-time control systems. Running an AI model next to the control loop enables
predictive maintenance and fault detection (arc fault, motor bearing fault, blower
imbalance) on the same device.

Tiny ML ModelZoo |release| supports two C2000 CPU generations:

* **F28x** devices with the 32-bit **C28x** DSP core (compiled with the TI C2000 code
  generation tools, ``cl2000``).
* **F29x** devices with the **C29x** DSP core (compiled with ``c29clang``).

F28x Devices (C28x)
-------------------

.. list-table::
   :header-rows: 1
   :widths: 14 10 44 32

   * - Device
     - NPU
     - Supported tasks
     - SDK example in C2000Ware
   * - **F28P55**
     - **Yes**
     - Classification (generic, arc fault, motor fault, blower imbalance), regression,
       forecasting, anomaly detection
     - ``f28p55x``
   * - F28P65
     - No
     - Same as F28P55
     - ``f28p65x``
   * - F28003
     - No
     - Classification (generic, arc fault, motor fault, blower imbalance, ECG),
       regression, forecasting, anomaly detection
     - ``f28003x``
   * - F28004
     - No
     - Same as F28003
     - ``f28004x``
   * - F2837
     - No
     - Same as F28003
     - --
   * - F28P551x
     - No
     - Same as F28003
     - --
   * - F2838x
     - No
     - Same as F28003
     - --
   * - F2837xS
     - No
     - Classification (generic, arc fault, motor fault, blower imbalance, ECG)
     - --
   * - F2807x
     - No
     - Classification (generic, arc fault, motor fault, blower imbalance, ECG)
     - --
   * - F28002x
     - No
     - Classification (motor fault, blower imbalance, ECG), regression, forecasting,
       anomaly detection
     - --
   * - F280013
     - No
     - Classification (arc fault, motor fault, blower imbalance, ECG), regression,
       forecasting, anomaly detection
     - --
   * - F280015
     - No
     - Same as F280013
     - --
   * - F28E12
     - No
     - Regression, forecasting, anomaly detection
     - --

F28P65 is compiled with 64-bit floating-point support (``--float_support=fpu64``); all
other F28x devices use ``fpu32``.

New in 1.5.0 for F28x
^^^^^^^^^^^^^^^^^^^^^

* **F28E12** is a new device profile. Its models are the regression, forecasting and
  anomaly-detection models; there is no classification model for F28E12 in 1.5.0.
* The Gen3 F28x devices **F2838x, F28P551x and F28002x** are now listed by the
  regression, forecasting and anomaly-detection models.
* Device lists were tightened to the verified per-model support. For example, F280013,
  F280015 and F28002x are not targets of ``generic_timeseries_classification``, and
  F2837xS and F2807x are classification-only.
* Feature-extraction memory is estimated for F28P55, F28P65, F2837, F28003, F28004,
  F280013, F280015 and F28E12 (see :doc:`device_overview`).

F28P55 (NPU)
^^^^^^^^^^^^

F28P55 is the C2000 device with the TI NPU. Use an ``_NPU`` model and keep the default
TINPU quantization so that the hardware NPU is used:

.. code-block:: yaml

   common:
     task_type: arc_fault
     target_device: F28P55
   dataset:
     dataset_name: dc_arc_fault_example_dsk
   training:
     model_name: CLS_1k_NPU
   compilation:
     compile_preset_name: default_preset    # or compress_npu_layer_data, forced_soft_npu_preset

This is the configuration of
`examples/dc_arc_fault/config_dsk.yaml <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/dc_arc_fault>`_
(with the preset spelled out). For the NPU rules see :doc:`npu_guidelines`.

On F28P55 the default preset uses the hardware NPU for classification, regression and
anomaly detection. Forecasting models are compiled for the CPU (``ti-npu type=soft``)
in 1.5.0.

Other F28x Devices
^^^^^^^^^^^^^^^^^^

All other F28x devices run the model on the C28x CPU. The same ``_NPU`` models can be
used; they are compiled for the CPU:

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F28E12
   training:
     model_name: REGR_2k_NPU

F29x Devices (C29x)
-------------------

.. list-table::
   :header-rows: 1
   :widths: 14 10 44 32

   * - Device
     - NPU
     - Supported tasks
     - SDK example in the F29H85x SDK
   * - F29H85
     - No
     - Classification (generic, arc fault, motor fault, blower imbalance), regression,
       forecasting, anomaly detection
     - ``f29h85x``
   * - F29P58
     - No
     - Classification (generic), regression, forecasting, anomaly detection
     - --
   * - F29P32
     - No
     - Classification (generic), regression, forecasting, anomaly detection
     - --

F29x devices run the model on the C29x CPU. Feature-extraction memory is estimated for
F29H85. Several examples ship an F29H85 configuration, for example
`grid_fault_detection <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/grid_fault_detection>`_,
`mosfet_temp_prediction <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/mosfet_temp_prediction>`_ and
`generic_timeseries_anomalydetection <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/generic_timeseries_anomalydetection>`_.

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F29H85

SDKs and Tools
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Devices
     - SDK (version used by the deployment guides)
     - Where the AI examples are
   * - F28x
     - `C2000Ware <https://www.ti.com/tool/C2000WARE>`_ 6.01.00.00
     - ``C2000Ware_6_01_00_00/libraries/ai/examples/<example>/<device>/``
   * - F29x
     - `F29H85X-SDK <https://www.ti.com/tool/download/F29H85X-SDK/>`_ 1.04
     - see :doc:`/DEPLOYING_TO_SDK_EXAMPLE`

LaunchPads used in the deployment guides: LAUNCHXL-F28P55X, LAUNCHXL-F28P65X,
LAUNCHXL-F280039C, LAUNCHXL-F280049C and the F29H85X-SOM-EVM.

You do not need the SDK to train and compile: ModelMaker uses the C2000 compilers found
through ``TOOLS_PATH`` (see :doc:`/installation/index`). You need the SDK and
`Code Composer Studio <https://www.ti.com/tool/CCSTUDIO>`_ to run the model on the board.

Typical Applications
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Application
     - Example
   * - DC arc fault detection
     - `dc_arc_fault <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/dc_arc_fault>`_
   * - Motor bearing fault and fan blower imbalance
     - `motor_bearing_fault <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/motor_bearing_fault>`_,
       `blower_imbalance <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/blower_imbalance>`_
   * - Torque and temperature estimation
     - `torque_measurement_regression <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/torque_measurement_regression>`_,
       `forecasting_pmsm_rotor_temp <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/forecasting_pmsm_rotor_temp>`_
   * - Grid fault detection (F29H85)
     - `grid_fault_detection <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/grid_fault_detection>`_

Next Steps
----------

* :doc:`npu_guidelines` for F28P55 model constraints
* :doc:`/DEPLOYING_TO_SDK_EXAMPLE` to run the model in a C2000Ware or F29H85x SDK example
* :doc:`/deployment/npu_device_deployment` and :doc:`/deployment/non_npu_deployment`
