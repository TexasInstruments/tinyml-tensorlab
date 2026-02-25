====================
Connectivity Devices
====================

TI's connectivity devices combine wireless communication with edge ML
capabilities for IoT applications.

Overview
--------

Connectivity devices enable:

* Wireless sensor networks with edge AI
* Smart home and building automation
* Industrial IoT with local inference
* Bluetooth and Sub-GHz sensing applications

These devices perform ML inference locally, reducing latency and
network bandwidth while maintaining wireless connectivity.

Supported Devices
-----------------

**SimpleLink CC27xx Family**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Device
     - Core
     - Features
   * - CC2755
     - Cortex-M33 (96 MHz)
     - Bluetooth 5.4 LE, Thread, Zigbee, Matter

**SimpleLink CC13xx Family**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Device
     - Core
     - Features
   * - CC1352
     - Cortex-M4 (48 MHz)
     - Sub-GHz, Bluetooth 5.0 LE, multi-protocol
   * - CC1354
     - Cortex-M33
     - Sub-GHz, Bluetooth 5.2 LE

**SimpleLink CC35xx Family**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Device
     - Core
     - Features
   * - CC35X1
     - Cortex-M4
     - Wi-Fi

CC2755
------

The CC2755 is a modern wireless MCU with strong ML capabilities.

**Key Features:**

* 96 MHz Arm Cortex-M33 core
* Bluetooth 5.4 LE Audio
* Thread and Zigbee support
* Matter protocol ready
* 1 MB Flash
* 256 KB SRAM
* TrustZone security

**ML Capabilities:**

* CPU-based inference (no NPU)
* Support for models up to ~10k parameters
* Suitable for classification and anomaly detection

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'CC2755'

   training:
     model_name: 'CLS_4k'  # Standard models

CC1352
------

Multi-band wireless MCU for Sub-GHz and Bluetooth applications.

**Key Features:**

* 48 MHz Arm Cortex-M4F core
* Sub-GHz radio (for long range)
* Bluetooth 5.0 LE
* 352 KB Flash
* 80 KB SRAM
* Ultra-low power

**ML Capabilities:**

* More constrained than CC2755
* Best for small models (<4k parameters)
* Ideal for simple classification tasks

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'CC1352'

   training:
     model_name: 'CLS_1k'  # Small models

CC1354
------

Sub-GHz and Bluetooth wireless MCU with Cortex-M33 core.

**Key Features:**

* Arm Cortex-M33 core
* Sub-GHz radio
* Bluetooth 5.2 LE
* Low power operation

**ML Capabilities:**

* CPU-based inference (no NPU)
* Suitable for small to medium models

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'CC1354'

   training:
     model_name: 'CLS_2k'

CC35X1
------

Wi-Fi enabled wireless MCU.

**Key Features:**

* Arm Cortex-M4 core
* Wi-Fi connectivity

**ML Capabilities:**

* CPU-based inference (no NPU)
* Suitable for small models

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'CC35X1'

   training:
     model_name: 'CLS_2k'

AM26x Family
------------

The AM26x family uses Arm Cortex-R5F cores for real-time industrial
applications with Ethernet connectivity.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Device
     - Cores
     - Features
   * - AM263
     - Quad R5F (400 MHz)
     - Industrial Ethernet, high performance
   * - AM263P
     - Quad R5F (400 MHz)
     - Enhanced security features
   * - AM261
     - Single R5F (400 MHz)
     - Cost-optimized industrial

**Key Features:**

* Real-time capable Cortex-R5F cores
* Industrial Ethernet (EtherCAT, PROFINET, EtherNet/IP)
* High-speed ADCs and PWMs
* Suitable for larger ML models

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'AM263'  # or AM263P, AM261

   training:
     model_name: 'CLS_6k'  # Can handle larger models

Typical Applications
--------------------

**Wireless Sensor Networks**

Deploy ML-enabled sensors with wireless backhaul:

.. code-block:: yaml

   # Vibration sensor with Bluetooth reporting
   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'CC2755'

   training:
     model_name: 'AD_2k'

Use cases:

* Structural health monitoring
* Environmental sensing
* Asset tracking with condition monitoring

**Smart Home/Building**

Local inference for privacy and responsiveness:

.. code-block:: yaml

   # Occupancy detection
   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'CC2755'

   training:
     model_name: 'CLS_2k'

Use cases:

* Occupancy sensing
* HVAC optimization
* Security systems

**Industrial IoT**

Edge inference with industrial protocols:

.. code-block:: yaml

   # Motor monitoring with Ethernet reporting
   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'AM263'

   dataset:
     dataset_name: 'motor_fault_classification_dsk'

   training:
     model_name: 'CLS_6k'

Use cases:

* Predictive maintenance
* Quality inspection
* Process anomaly detection

**Long-Range IoT (Sub-GHz)**

Remote sensing with minimal power:

.. code-block:: yaml

   # Remote vibration sensor
   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'CC1352'

   training:
     model_name: 'AD_500'  # Minimal model

Power Optimization
------------------

Connectivity devices often run on batteries:

**Duty Cycling**

Run inference periodically, sleep between:

* Wake on timer or sensor threshold
* Perform inference
* Transmit only if anomaly detected
* Return to sleep

**Model Size vs Battery Life**

Smaller models use less energy per inference:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Model Size
     - Inference Energy
     - Battery Impact
   * - 500 params
     - ~10 µJ
     - Minimal
   * - 2k params
     - ~50 µJ
     - Low
   * - 4k params
     - ~150 µJ
     - Moderate

**Transmission Optimization**

* Send only alerts, not raw data
* Batch non-urgent communications
* Use lowest sufficient TX power

Memory Constraints
------------------

Connectivity devices have varying memory:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Device
     - Flash
     - RAM
     - Recommended Model
   * - CC2755
     - 1 MB
     - 256 KB
     - Up to CLS_6k
   * - CC1352
     - 352 KB
     - 80 KB
     - Up to CLS_2k
   * - CC1354
     - 1 MB
     - 256 KB
     - Up to CLS_4k
   * - CC35X1
     - 512 KB
     - 128 KB
     - Up to CLS_2k
   * - AM263
     - 2 MB
     - 512 KB
     - Up to CLS_13k

**Note:** Wireless stack consumes significant memory. Plan model size
accordingly.

Development Tools
-----------------

**Code Composer Studio (CCS)**

* Install SimpleLink SDK for CC27xx/CC13xx
* Install AM26x SDK for AM26x devices
* Use SysConfig for wireless stack configuration

**SimpleLink SDK**

* Wireless protocol stacks
* Example applications
* Power management

**TI 15.4-Stack**

For Sub-GHz mesh networks with CC13xx.

**Industrial Communications SDK**

For AM26x industrial Ethernet protocols.

Wireless Protocol Considerations
--------------------------------

Choose protocol based on application needs:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Protocol
     - Device
     - Best For
   * - Bluetooth LE
     - CC2755, CC1352, CC1354
     - Short range, smartphone integration
   * - Thread/Zigbee
     - CC2755
     - Home automation mesh
   * - Sub-GHz
     - CC1352, CC1354
     - Long range, building penetration
   * - Wi-Fi
     - CC35X1
     - IP connectivity, cloud integration
   * - Industrial Ethernet
     - AM26x
     - Factory automation, real-time

Getting Started
---------------

1. Choose device based on connectivity needs
2. Install appropriate SDK
3. Account for wireless stack memory usage
4. Select model size within remaining memory
5. Test inference + communication together

.. code-block:: yaml

   # Example: BLE sensor node
   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'CC2755'

   dataset:
     dataset_name: 'your_sensor_dataset'

   training:
     model_name: 'AD_2k'
     training_epochs: 30

   compilation:
     enable: True

Next Steps
----------

* See :doc:`device_overview` for complete device list
* Read :doc:`/deployment/non_npu_deployment` for CPU inference
* Explore :doc:`/task_types/anomaly_detection` for sensor monitoring
