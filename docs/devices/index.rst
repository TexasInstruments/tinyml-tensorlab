=================
Supported Devices
=================

Tiny ML ModelZoo |release| can train and compile models for **33 target devices**
(``common.target_device`` values) across nine TI device families. This section helps
you find your device, see what it supports, and choose the right compilation preset.

.. note::

   Device support in 1.5.0 is **derived from the models**: a device is offered for a
   task only if at least one model for that task lists the device in its
   ``target_devices``. A device that has no model for your task is rejected by
   ModelMaker with a "task_type ... is not supported for device ..." error. The full
   per-device matrix is in :doc:`device_overview`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   device_overview
   npu_guidelines
   /NPU_CONFIGURATION_GUIDELINES
   c2000_family
   mspm0_family
   connectivity_devices

Device Families at a Glance
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 30 18 30

   * - Family
     - ``target_device`` values
     - CPU core
     - Hardware NPU
   * - :doc:`C2000 F28x <c2000_family>`
     - F28P55, F28P65, F28P551x, F2837, F2837xS, F2838x, F2807x, F28003, F28004,
       F28002x, F280013, F280015, F28E12
     - TI C28x DSP
     - F28P55
   * - :doc:`C2000 F29x <c2000_family>`
     - F29H85, F29P58, F29P32
     - TI C29x DSP
     - None
   * - :doc:`MSPM0 <mspm0_family>`
     - MSPM0G3507, MSPM0G3519, MSPM0G5187
     - Arm Cortex-M0+
     - MSPM0G5187
   * - :doc:`MSPM33C <connectivity_devices>`
     - MSPM33C32, MSPM33C34
     - Arm Cortex-M33
     - MSPM33C34 (no model support in 1.5.0)
   * - :doc:`AM13 <device_overview>`
     - AM13E2
     - Arm Cortex-M33
     - AM13E2
   * - :doc:`AM26x <connectivity_devices>`
     - AM263, AM263P, AM261
     - Arm Cortex-R5F
     - None
   * - :doc:`Connectivity <connectivity_devices>`
     - CC2755, CC2745, CC1312, CC1314, CC1352, CC1354, CC35X1
     - Arm Cortex-M33 / Cortex-M4
     - None (CC2755, CC2745, CC35X1 use the Arm Custom Datapath Extension)
   * - :doc:`Radar <connectivity_devices>`
     - xWRL6432
     - Arm Cortex-M4
     - None

What's New for Devices in 1.5.0
-------------------------------

* **Gen3 F28x devices reconciled.** F2838x, F28P551x, F28002x and the new **F28E12**
  device profile are now listed by the regression, forecasting and anomaly-detection
  models. Over-broad device lists (for example arc fault and generic time series
  classification) were tightened to the devices each model was verified on.
* **AM13E2 image classification.** MobileNet-style image classifiers were compressed
  to fit the AM13 memory budget, and AM13E2 is now a target for
  ``image_classification``.
* **Radar.** The radar tasks (point cloud and surface classification) target the xWRL6432 radar sensor.
* **Feature-extraction memory estimation.** For AM13E2, MSPM0G3507, MSPM0G5187,
  F29H85 and many C28x devices, ModelMaker compiles the SDK feature-extraction library and reports its
  Code/RO/RW memory next to the model memory. See :doc:`device_overview`.
* **Derived device lists.** Task-level device lists for radar, image and audio
  classification are computed from each model's real device support, so a task no
  longer advertises devices that no model supports.

Where to Go Next
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Device Overview
      :link: device_overview
      :link-type: doc

      All 33 devices with core, NPU, supported tasks, SDK example and memory estimation.

   .. grid-item-card:: NPU Guidelines
      :link: npu_guidelines
      :link-type: doc

      Which devices have an NPU, the compilation presets, and the key layer rules.

   .. grid-item-card:: NPU Configuration Guidelines
      :link: /NPU_CONFIGURATION_GUIDELINES
      :link-type: doc

      The detailed layer-by-layer constraints for NPU-compatible models.

   .. grid-item-card:: Deploy to a Device
      :link: /deployment/index
      :link-type: doc

      Copy the compiled model and golden vectors into an SDK example and run it.
