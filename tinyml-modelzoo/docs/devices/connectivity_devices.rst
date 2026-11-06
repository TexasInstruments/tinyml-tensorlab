==============================================
Connectivity, AM26x, MSPM33 and Radar Devices
==============================================

This page covers the device families that do not have a page of their own: the SimpleLink
connectivity MCUs (CC27xx, CC13xx, CC35xx), the AM26x Arm Cortex-R5F family, the MSPM33C
family, and the xWRL6432 radar sensor. For the per-device task matrix, SDK example and
memory-estimation support see :doc:`device_overview`. AM13E2 is listed in
:doc:`device_overview` as well; it has a hardware NPU (see :doc:`npu_guidelines`).

.. note::

   The ``tinyml-sdk`` repository ships on-device AI examples for C2000 (c28, c29), AM13
   and AM26 only. There are **no** Tiny ML ModelZoo SDK examples for the CC,
   MSPM33 or xWRL6432 devices. For those devices ModelMaker produces the compiled model
   library and golden vectors, and you integrate them into your own SimpleLink / MSPM33 /
   mmWave project. The hardware numbers for the CC devices in the 1.4.0 guide
   (flash, RAM, energy per inference, recommended model sizes) could not be verified against
   1.5.0 sources and are not repeated here; use the device data sheets.

Connectivity Devices (CC)
-------------------------

No CC device has a hardware NPU (only F28P55, MSPM0G5187, AM13E2 and MSPM33C34 do). They use
one of two compilation styles, set in ``tinyml-modelmaker`` (``timeseries/constants.py``).

.. list-table::
   :header-rows: 1
   :widths: 18 24 58

   * - ``target_device``
     - Compiler core flags
     - Compilation style
   * - CC2755, CC2745
     - Cortex-M33, ``thumbv8.1-m.main+cdecp0``
     - Default preset uses the Arm Custom Datapath Extension (CDE), ``ti-npu type=m33cde``.
       A software-only ("soft") preset is also defined.
   * - CC35X1
     - Cortex-M33, ``thumbv8.1-m.main+cdecp0``
     - Same as CC2755: CDE preset by default, soft preset available.
   * - CC1354, CC1314
     - Cortex-M33, ``armv8-m.main``, FPv5-SP
     - Software (soft) preset only.
   * - CC1352, CC1312
     - Cortex-M4F, ``armv7e-m``, FPv4-SP
     - Software (soft) preset only.

.. note::

   The 1.4.0 guide listed CC2755, CC1352, CC1354 and CC35X1 (as a Cortex-M4 Wi-Fi part). In
   1.5.0 the device list also contains **CC2745, CC1312 and CC1314**, and the compiler
   flags for CC35X1 are those of a Cortex-M33 with CDE. The CC devices are listed by
   classification, motor fault and PIR detection models (see :doc:`device_overview`);
   regression, forecasting and anomaly-detection models do not list them.

Example configuration:

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'CC2755'

   compilation:
     enable: True

Choose a model that lists your device; ModelMaker rejects a model that does not.

AM26x (AM263, AM263P, AM261)
----------------------------

Arm Cortex-R5F devices (VFPv3-D16 floating point), compiled without an NPU. AM263 supports
classification, arc fault, motor fault, blower imbalance, regression, forecasting and anomaly
detection models; AM263P and AM261 support classification, regression, forecasting and anomaly
detection (see :doc:`device_overview`). The ``tinyml-sdk`` folder ``am26/ai/examples``
contains the same twelve example projects as the AM13 folder (arc_fault, blower_imbalance,
fan_blade_anomalydetection, forecasting_pmsm_rotor_temp, generic_timeseries_*,
hvac_indoor_temp_forecast, motor_fault, torque_measurement, washing_machine_load_weighing).
Deployment follows :doc:`/deployment/non_npu_deployment`.

MSPM33C (MSPM33C32, MSPM33C34)
------------------------------

Arm Cortex-M33 devices. **MSPM33C32** is a CPU-only target (MSPM33 SDK ``mspm33c321a``).
**MSPM33C34** has an NPU compilation profile and is accepted on the command line, but no
1.5.0 model lists it, so every task is rejected for it. Use MSPM33C32 for MSPM33 projects.

xWRL6432 (Radar)
----------------

The xWRL6432 mmWave radar sensor (Arm Cortex-M4F, no NPU) is the target of the
radar tasks (point cloud and surface classification) and the only device they list.

Where to Go Next
----------------

* :doc:`device_overview` for the complete device matrix
* :doc:`/deployment/non_npu_deployment` for CPU-only deployment
* :doc:`/deployment/index` for the deployment guides
