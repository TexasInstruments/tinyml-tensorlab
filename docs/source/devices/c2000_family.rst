============
C2000 Family
============

The C2000 family of digital signal processors (DSPs) is TI's flagship platform
for real-time industrial control applications.

Overview
--------

C2000 devices feature:

* High-performance DSP cores optimized for control algorithms
* Integrated analog peripherals (ADCs, comparators)
* PWM modules for motor control
* Rich communication interfaces

These capabilities make C2000 ideal for predictive maintenance and real-time
fault detection where ML inference runs alongside control loops.

Supported Devices
-----------------

**NPU-Enabled**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Device
     - Core
     - Frequency
     - Features
   * - **F28P55**
     - C28x + NPU
     - 150 MHz
     - TINPU neural accelerator, ideal for complex models

**C28x Core Devices**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Device
     - Core
     - Frequency
     - Features
   * - F28P65
     - C28x
     - 150 MHz
     - High-performance, no NPU
   * - F2837
     - Dual C28x
     - 200 MHz
     - Dual-core, legacy support
   * - F28004
     - C28x
     - 100 MHz
     - Mid-range, motor control
   * - F28003
     - C28x
     - 100 MHz
     - Cost-optimized
   * - F280013
     - C28x
     - 100 MHz
     - Entry-level
   * - F280015
     - C28x
     - 120 MHz
     - Enhanced entry-level

**C29x Core Devices (64-bit)**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Device
     - Core
     - Frequency
     - Features
   * - F29H85
     - C29x
     - 300 MHz
     - 64-bit, high performance
   * - F29P58
     - C29x
     - 300 MHz
     - 64-bit, balanced
   * - F29P32
     - C29x
     - 300 MHz
     - 64-bit, cost-optimized

F28P55 (Recommended)
--------------------

The F28P55 is the recommended device for Tiny ML applications due to its
integrated TINPU (Neural Processing Unit).

**Key Specifications:**

* 150 MHz C28x DSP core
* TINPU neural accelerator
* 512 KB Flash
* 128 KB RAM
* 16-bit and 12-bit ADCs
* Type-4 CLB (Configurable Logic Block)

**NPU Capabilities:**

* 8-bit and 4-bit quantized inference
* Hardware-accelerated convolutions
* Up to 25x faster than CPU inference
* Support for models up to ~60k parameters

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'F28P55'

   training:
     model_name: 'CLS_4k_NPU'  # Use NPU-compatible models

   compilation:
     preset_name: 'compress_npu_layer_data'

F28P65
------

The F28P65 offers similar performance to F28P55 but without the NPU.

**When to Use:**

* Existing F28P65 designs
* Models that don't fit NPU constraints
* Projects not requiring fastest inference

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'F28P65'

   training:
     model_name: 'CLS_4k'  # Standard models (no _NPU suffix)

F2837
-----

Legacy dual-core device for existing systems.

**Features:**

* Dual C28x cores at 200 MHz
* Large memory (1 MB Flash)
* CLA (Control Law Accelerator)

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'F2837'

C29x Family (F29H85, F29P58, F29P32)
------------------------------------

The newest C2000 generation with 64-bit C29x cores.

**Features:**

* 64-bit native architecture
* 300 MHz operation
* Enhanced floating-point performance
* Backward compatibility with C28x code

**Configuration:**

.. code-block:: yaml

   common:
     target_device: 'F29H85'  # or F29P58, F29P32

Typical Applications
--------------------

C2000 devices excel in these ML applications:

**Motor Fault Detection**

* Bearing fault classification
* Winding fault detection
* Vibration anomaly detection

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'motor_fault_classification_dsk'

**Arc Fault Detection**

* DC arc fault in solar/battery systems
* AC arc fault in building wiring

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'

**Power Quality Monitoring**

* Harmonic analysis
* Transient detection
* Grid fault classification

Development Tools
-----------------

**Code Composer Studio (CCS)**

TI's official IDE for C2000 development:

* Download from: https://www.ti.com/tool/CCSTUDIO
* Version 12.x or later recommended
* Install C2000 device support

**C2000WARE SDK**

Software development kit with:

* Device drivers
* Example projects
* Documentation

**controlSUITE**

Legacy SDK (use C2000WARE for new projects)

Memory Considerations
---------------------

C2000 devices have different memory architectures:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - Device
     - Flash
     - RAM
     - Max Model Size
   * - F28P55
     - 512 KB
     - 128 KB
     - ~60k params (NPU)
   * - F28P65
     - 512 KB
     - 128 KB
     - ~20k params (CPU)
   * - F2837
     - 1 MB
     - 200 KB
     - ~30k params (CPU)

Choose model sizes that fit within device constraints:

.. code-block:: yaml

   # F28P55 with NPU - can use larger models
   training:
     model_name: 'CLS_13k_NPU'

   # F28004 - use smaller models
   training:
     model_name: 'CLS_1k'

Next Steps
----------

* Review :doc:`npu_guidelines` for F28P55 model constraints
* See :doc:`/deployment/ccs_integration` for device programming
* Try :doc:`/examples/arc_fault` example on C2000
