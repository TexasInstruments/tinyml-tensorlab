===============
Device Overview
===============

Tiny ML Tensorlab supports over 20 Texas Instruments microcontrollers across
multiple device families.

Supported Device Families
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Family
     - Core
     - Applications
   * - **C2000 DSP**
     - TI C28x/C29x
     - Industrial control, motor drives, power systems
   * - **MSPM0**
     - Arm Cortex-M0+
     - Ultra-low power, cost-sensitive IoT
   * - **MSPM33C**
     - Arm Cortex-M33
     - Security-critical, high-performance edge
   * - **AM13**
     - Arm Cortex-M33
     - High-performance edge with NPU acceleration
   * - **AM26x**
     - Arm Cortex-R5
     - Industrial Ethernet, real-time systems
   * - **Connectivity**
     - Arm Cortex-M33/M4
     - Wireless IoT, Bluetooth, Sub-GHz

Complete Device List
--------------------

**C2000 Family**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Device
     - NPU
     - Description
   * - **F28P55**
     - Yes
     - Recommended for complex models. 32-bit, 150 MHz, NPU-accelerated
   * - F28P65
     - No
     - 32-bit, 150 MHz
   * - F29H85
     - No
     - 64-bit C29x core
   * - F29P58
     - No
     - 64-bit C29x core
   * - F29P32
     - No
     - 64-bit C29x core
   * - F2837
     - No
     - Dual-core, 200 MHz
   * - F28003
     - No
     - 100 MHz
   * - F28004
     - No
     - 100 MHz
   * - F280013
     - No
     - 100 MHz
   * - F280015
     - No
     - 120 MHz

**MSPM0 Family**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Device
     - NPU
     - Description
   * - MSPM0G3507
     - No
     - 80 MHz, ultra-low power
   * - MSPM0G3519
     - No
     - 80 MHz, ultra-low power
   * - **MSPM0G5187**
     - Yes
     - 80 MHz, NPU-accelerated

**MSPM33C Family**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Device
     - NPU
     - Description
   * - MSPM33C32
     - No
     - 160 MHz, TrustZone security
   * - MSPM33C34
     - No
     - 160 MHz, extended peripherals

**AM13 Family**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Device
     - NPU
     - Description
   * - **AM13E2**
     - Yes
     - 160 MHz Arm Cortex-M33, NPU-accelerated, TrustZone security

**AM26x Family**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Device
     - NPU
     - Description
   * - AM263
     - No
     - Quad-core Cortex-R5F, 400 MHz
   * - AM263P
     - No
     - Quad-core Cortex-R5F, 400 MHz
   * - AM261
     - No
     - Single-core Cortex-R5F, 400 MHz

**Connectivity Devices**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Device
     - NPU
     - Description
   * - CC2755
     - No
     - 96 MHz Cortex-M33, wireless
   * - CC1352
     - No
     - Cortex-M4, multi-protocol wireless

Target Device Configuration
---------------------------

Specify your target device in the config:

.. code-block:: yaml

   common:
     target_device: 'F28P55'  # or 'MSPM0G3507', 'AM263', etc.

NPU vs Non-NPU Devices
----------------------

**NPU Devices (F28P55, AM13E2, MSPM0G5187)**:

* Hardware-accelerated inference
* Faster execution (10-100x speedup)
* Lower power consumption
* Requires NPU-compatible models (``*_NPU`` variants)
* Specific layer constraints (see :doc:`npu_guidelines`)

**Non-NPU Devices**:

* Software-only inference
* More flexible model architectures
* Suitable for simpler models
* Use standard model variants

Choosing a Device
-----------------

Consider these factors:

1. **Model Complexity**

   * Simple models (<1K params): Any device
   * Medium models (1K-10K params): Mid-range devices
   * Complex models (>10K params): NPU devices recommended

2. **Latency Requirements**

   * Real-time (<1ms): NPU devices
   * Near real-time (<10ms): Most devices
   * Relaxed timing: Any device

3. **Power Budget**

   * Battery-powered: MSPM0 family
   * Always-on: C2000, AM26x families

4. **Existing Infrastructure**

   * Motor control: C2000 family
   * Industrial Ethernet: AM26x family
   * Wireless: CC27xx, CC13xx
