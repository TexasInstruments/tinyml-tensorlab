=============
PIR Detection
=============

Detect presence and motion using PIR sensor data.

Overview
--------

This example demonstrates presence and motion detection using Passive Infrared
(PIR) sensor data. Unlike simple threshold-based detection, this ML approach
can distinguish between different types of motion and reduce false positives.

**Application**: Security systems, occupancy sensing, smart lighting, IoT

**Task Type**: Time Series Classification

**Data Type**: Multivariate (PIR sensor readings)

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'CC2755'  # Optimized for wireless PIR applications

   dataset:
     dataset_name: 'pir_detection'

   training:
     model_name: 'PIRDetection_model_1_t'
     training_epochs: 50
     batch_size: 32

   testing: {}
   compilation: {}

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/pir_detection/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\pir_detection\config.yaml

Dataset Details
---------------

**Input Variables**:

* PIR sensor analog readings
* Multiple PIR channels (if available)

**Classes**:

* No presence
* Presence detected
* Motion detected (optional sub-classes)

Recommended Devices
-------------------

This example is optimized for TI's connectivity devices:

* **CC2755**: Wireless MCU with Cortex-M33
* **CC1352**: Multi-protocol wireless MCU

See Also
--------

* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/pir_detection>`_
