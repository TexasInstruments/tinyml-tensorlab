====================
Human Fall Detection
====================

.. _example-fall-detection-classification:

Edge AI solution for real-time human fall detection from accelerometer data on the MSPM0G5187 with NPU
acceleration. For the radar based variant, see the radar examples.

Example folder: ``examples/fall_detection_classification``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/fall_detection_classification>`_)

Overview
--------

The application classifies human movement as Activities of Daily Living (ADL) or Fall. Falls are a
leading cause of injury among elderly people and in industrial environments, and traditional systems rely on
cloud connectivity or bulky wearables. Here the classification runs directly on a microcontroller, for
example in a belt-mounted safety device.

* **Application**: elderly care, worker safety, wearable safety systems
* **Task**: binary classification (``generic_timeseries_classification``): ADL or Fall
* **Data**: 3-axis accelerometer (``InputChannel_3``)
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config)
* **Model**: ``CLS_6k_NPU``
* **Targets**: real-time classification, accuracy of at least 97.5%, low memory footprint

System Components
-----------------

* Hardware: `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ and the EdgeAI Boosterpack
  `TIDA-010997 <https://www.ti.com/tool/TIDA-010997>`_ with a BMI270 accelerometer
* Software: Code Composer Studio 12.x or later, MSPM0 SDK 2.11.00 or later

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/fall_detection_classification/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\fall_detection_classification\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: fall_detection_classification
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/fall_detection_classification.zip
   data_processing_feature_extraction:
     feature_extraction_name: FallDetection_256Input_FE_RFFT_8Feature_8Frame_3InputChannel_removeDC_2D1
   training:
     model_name: CLS_6k_NPU
     batch_size: 256
     training_epochs: 50
     num_gpus: 0
   testing: {}
   compilation: {}

Dataset
-------

The example uses the `SisFall <https://www.mdpi.com/1424-8220/17/1/198>`_ dataset (about 4500 recordings, two
classes). Each SisFall file has three sensors: ADXL345 (13-bit accelerometer), ITG3200 (gyroscope) and MMA8451Q
(accelerometer). The prepared zip is downloaded automatically. The preparation applied to it was:

#. **Sensor filtering**: only the ADXL345 accelerometer columns are kept, to match the single accelerometer
   (BMI270) of the target hardware.
#. **Bit-depth scaling**: the 13-bit ADXL345 values are multiplied by 8 (2^16 / 2^13) so that they match the
   16-bit BMI270 data seen by the firmware during inference.
#. **Format conversion**: each file is parsed, labeled ADL or Fall and exported in the ModelZoo dataset structure.

Feature Extraction
------------------

The preset ``FallDetection_256Input_FE_RFFT_8Feature_8Frame_3InputChannel_removeDC_2D1`` mirrors the on-device
pipeline:

1. 3-axis accelerometer input scaled to 16 bit
2. 256-point real FFT (ARM CMSIS-DSP on device)
3. Complex magnitude
4. DC removal
5. Binning: 16 adjacent bins averaged into 8 features
6. Concatenation of 8 frames (64 features per axis)

Model and Results
-----------------

The example readme reports for the reference model (``CLS_6k``, about 6,000 parameters), measured on the
LP-MSPM0G5187 LaunchPad with an earlier release: 14 kB flash, 0.7 kB RAM, 0.67 ms NPU inference latency and 97.65%
accuracy, with INT8 quantization.

In the automated 1.5.0 config test run, ``config_MSPM0.yaml`` (``CLS_6k_NPU``, 6,696 parameters) reached a test
accuracy of 97.15% with an AUC ROC of 0.996.

Practical Considerations
------------------------

* Mount the accelerometer on the belt or wrist and keep the sensor firmly attached.
* The BMI270 sampling rate is 100 to 200 Hz.
* False positives (rapid ADL movements such as jumping) can be reduced by adding such samples to the training data.
* All inference runs on the device; raw accelerometer data never leaves it.

Deployment
----------

See :doc:`/deploying_classification_models_from_modelmaker_to_device/readme` and
:doc:`/DEPLOYING_TO_SDK_EXAMPLE`. To adapt the model to a person after deployment, see
:doc:`/ondevice_training/overview`.

See Also
--------

* :doc:`pir_detection`: another sensor classification example on MSPM0
* :doc:`har_activity_recognition`: activity recognition from accelerometer data
