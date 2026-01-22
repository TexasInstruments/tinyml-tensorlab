=============
MSPM0 Family
=============

The MSPM0 family features Arm Cortex-M0+ processors optimized for
ultra-low power and cost-sensitive applications.

Overview
--------

MSPM0 devices are designed for:

* Battery-powered IoT devices
* Cost-sensitive consumer products
* Always-on sensing applications
* Simple predictive maintenance

The Cortex-M0+ core provides an excellent balance of performance and
power efficiency for edge ML applications.

Supported Devices
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Device
     - NPU
     - Frequency
     - Features
   * - MSPM0G3507
     - No
     - 80 MHz
     - Ultra-low power, cost-optimized
   * - MSPM0G3519
     - No
     - 80 MHz
     - Enhanced peripherals
   * - **MSPM0G5187**
     - **Yes**
     - 80 MHz
     - NPU-accelerated inference

MSPM0G5187 (NPU-Enabled)
------------------------

The MSPM0G5187 is the recommended device for Tiny ML on MSPM0.

**Key Features:**

* 80 MHz Arm Cortex-M0+ core
* TINPU neural accelerator
* 256 KB Flash
* 64 KB SRAM
* Ultra-low power modes
* Integrated analog (12-bit ADC, DAC, comparators)

**NPU Capabilities:**

* 8-bit quantized inference
* Hardware convolution acceleration
* Significantly faster than CPU-only inference
* Optimized for small models (up to ~10k parameters)

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'MSPM0G5187'

   training:
     model_name: 'CLS_1k_NPU'  # NPU-compatible models

   compilation:
     preset_name: 'compress_npu_layer_data'

MSPM0G3507
----------

Entry-level device without NPU, suitable for simpler models.

**Key Features:**

* 80 MHz Cortex-M0+ core
* 128 KB Flash
* 32 KB SRAM
* Low cost
* Rich analog integration

**Best For:**

* Simple classification tasks
* Binary anomaly detection
* Cost-constrained applications

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'MSPM0G3507'

   training:
     model_name: 'CLS_100'  # Very small models

MSPM0G3519
----------

Enhanced variant with additional peripherals.

**Additional Features:**

* More GPIO pins
* Additional communication interfaces
* Extended temperature range options

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'MSPM0G3519'

AM13 Family
-----------

The AM13 family is a separate device family featuring Arm Cortex-M33 cores
with NPU acceleration for high-performance edge ML applications.

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Device
     - NPU
     - Frequency
     - Features
   * - **AM13E2**
     - **Yes**
     - 160 MHz
     - NPU-accelerated, TrustZone security

**AM13E2 (NPU-Enabled):**

The AM13E2 combines Cortex-M33 performance with NPU acceleration, making it
ideal for security-critical and high-performance edge ML applications.

**Key Features:**

* 160 MHz Arm Cortex-M33 core
* TINPU neural accelerator
* TrustZone security
* Higher performance than MSPM0 family

.. code-block:: yaml

   common:
     target_device: 'AM13E2'

   training:
     model_name: 'CLS_4k_NPU'

   compilation:
     preset_name: 'compress_npu_layer_data'

Power Considerations
--------------------

MSPM0 devices excel in low-power applications:

**Active Power:**

* ~100 µA/MHz typical
* Ideal for continuous sensing

**Sleep Modes:**

* Standby: ~1-5 µA
* Shutdown: <100 nA

**Design Tips:**

1. Use smallest model that meets accuracy needs
2. Batch inferences where possible
3. Use wake-on-event for sparse data
4. Consider inference duty cycling

Memory Constraints
------------------

MSPM0 devices have limited memory compared to C2000:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Device
     - Flash
     - SRAM
     - Recommended Model
   * - MSPM0G3507
     - 128 KB
     - 32 KB
     - CLS_100 to CLS_500
   * - MSPM0G3519
     - 128 KB
     - 32 KB
     - CLS_100 to CLS_500
   * - MSPM0G5187
     - 256 KB
     - 64 KB
     - CLS_1k_NPU to CLS_4k_NPU

**Memory Optimization:**

.. code-block:: yaml

   # Use smaller feature extraction
   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_FFTBIN_32Feature_4Frame'

   # Use quantized models
   training:
     model_name: 'CLS_500_NPU'
     quantization_type: 'int8'

Typical Applications
--------------------

**Wearable Devices:**

* Activity classification
* Gesture recognition
* Health monitoring

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'MSPM0G5187'

   training:
     model_name: 'CLS_1k_NPU'

**Smart Sensors:**

* Vibration anomaly detection
* Environmental monitoring
* Acoustic event detection

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'MSPM0G5187'

   training:
     model_name: 'AD_1k_NPU'

**Consumer Electronics:**

* Voice activity detection
* Simple keyword spotting
* Touch gesture classification

Development Tools
-----------------

**Code Composer Studio (CCS)**

* Install MSPM0 device support
* Use SysConfig for pin configuration
* Built-in debug support

**MSPM0 SDK**

* Peripheral drivers
* Example projects
* Power management libraries

**LaunchPad Development Kits**

* LP-MSPM0G3507: Entry-level evaluation
* LP-MSPM0G5187: NPU evaluation (when available)

Getting Started
---------------

1. Install CCS with MSPM0 support
2. Install MSPM0 SDK
3. Choose appropriate model size for your device
4. Train with NPU-compatible model (for MSPM0G5187)
5. Deploy using CCS

.. code-block:: yaml

   # Example configuration for MSPM0G5187
   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'MSPM0G5187'

   dataset:
     dataset_name: 'hello_world_example_dsg'

   training:
     model_name: 'CLS_1k_NPU'
     training_epochs: 20

   compilation:
     enable: True

Next Steps
----------

* Review :doc:`npu_guidelines` for MSPM0G5187/AM13E2
* See :doc:`/examples/hello_world` for a simple starting point
* Read :doc:`/deployment/npu_device_deployment` for deployment
