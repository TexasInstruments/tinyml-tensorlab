=================
Supported Devices
=================

Tiny ML Tensorlab supports over 20 Texas Instruments microcontrollers across
multiple device families. This section helps you understand device capabilities
and choose the right MCU for your application.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   device_overview
   npu_guidelines
   c2000_family
   mspm0_family
   connectivity_devices

Device Families at a Glance
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Family
     - NPU Support
     - Architecture
     - Best For
   * - **C2000 DSP**
     - F28P55 only
     - 32/64-bit
     - Industrial control, motor drives, power systems
   * - **MSPM0**
     - MSPM0G5187
     - Arm Cortex-M0+
     - Ultra-low power, cost-sensitive applications
   * - **MSPM33C**
     - None
     - Arm Cortex-M33
     - Security-critical, high-performance edge
   * - **AM13**
     - AM13E2
     - Arm Cortex-M33
     - High-performance edge with NPU acceleration
   * - **AM26x**
     - None
     - Arm Cortex-R5
     - Industrial Ethernet, high-reliability systems
   * - **Connectivity**
     - None
     - Arm Cortex-M33/M4
     - Wireless IoT, connected sensors

Complete Device List
--------------------

**C2000 Family:**

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Device
     - NPU
     - Description
   * - **F28P55**
     - Yes
     - 32-bit C28x, 150 MHz, NPU-accelerated (recommended for complex models)
   * - F28P65
     - No
     - 32-bit C28x, 150 MHz
   * - F29H85
     - No
     - 64-bit C29x core, high performance
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
     - 100 MHz, cost-effective
   * - F28004
     - No
     - 100 MHz, cost-effective
   * - F280013
     - No
     - 100 MHz, entry-level
   * - F280015
     - No
     - 120 MHz, entry-level

**MSPM0 Family:**

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Device
     - NPU
     - Description
   * - **MSPM0G5187**
     - Yes
     - 80 MHz Cortex-M0+, NPU-accelerated, ultra-low power
   * - MSPM0G3507
     - No
     - 80 MHz Cortex-M0+, ultra-low power
   * - MSPM0G3519
     - No
     - 80 MHz Cortex-M0+, ultra-low power

**MSPM33C Family:**

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Device
     - NPU
     - Description
   * - MSPM33C32
     - No
     - 160 MHz Cortex-M33, TrustZone security
   * - MSPM33C34
     - No
     - 160 MHz Cortex-M33, extended peripherals

**AM13 Family:**

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Device
     - NPU
     - Description
   * - **AM13E2**
     - Yes
     - 160 MHz Cortex-M33, NPU-accelerated, TrustZone security

**AM26x Family:**

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Device
     - NPU
     - Description
   * - AM263
     - No
     - Quad-core Cortex-R5F, 400 MHz, industrial Ethernet
   * - AM263P
     - No
     - Quad-core Cortex-R5F, 400 MHz, enhanced peripherals
   * - AM261
     - No
     - Single-core Cortex-R5F, 400 MHz

**Connectivity Devices:**

.. list-table::
   :header-rows: 1
   :widths: 20 10 70

   * - Device
     - NPU
     - Description
   * - CC2755
     - No
     - 96 MHz Cortex-M33, Wi-Fi/Bluetooth
   * - CC1352
     - No
     - Cortex-M4, multi-protocol wireless (Sub-GHz, BLE)
   * - CC1354
     - No
     - Cortex-M33, Sub-GHz + Bluetooth 5.2 LE
   * - CC35X1
     - No
     - Cortex-M4, Wi-Fi

NPU vs Non-NPU Devices
----------------------

**NPU-Enabled Devices** (F28P55, AM13E2, MSPM0G5187):

* Hardware-accelerated neural network inference
* Faster execution, lower power consumption
* Requires NPU-compatible model architectures
* Best for complex models with strict latency requirements

**Non-NPU Devices**:

* Software-based neural network execution
* More flexibility in model architecture
* Suitable for simpler models or less time-critical applications
* Lower cost options available

See :doc:`npu_guidelines` for detailed information on designing NPU-compatible models.
