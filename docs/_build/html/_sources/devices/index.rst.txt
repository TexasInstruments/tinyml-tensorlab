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
