================================
Dynamic Hand Gesture Recognition
================================

Real-time dynamic hand gesture classification from 3-axis accelerometer data on MSPM0G5187 with integrated NPU.

Overview
--------

* **Task**: Multi-class classification (4 gesture types)
* **Application**: Gesture-based human-machine interface
* **Dataset**: 3-axis accelerometer recordings from TI Sensor BoosterPack
* **Device**: MSPM0G5187 (NPU-accelerated)

Gesture Classes
---------------

1. **Circle** — Clockwise or counter-clockwise circular motion
2. **Wave** — Side-to-side waving motion
3. **Tap** — Short impact gesture
4. **Others** — Non-gesture or unrecognized motion

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/dynamic_hand_gesture_recognition/config_MSPM0.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\dynamic_hand_gesture_recognition\config_MSPM0.yaml

Device Support
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 50 10

   * - Device
     - Hardware
     - Configuration File
   * - ``MSPM0G5187``
     - MSPM0 with NPU + TI Sensor BoosterPack (3-axis accelerometer)
     - ``config_MSPM0.yaml``

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'MSPM0G5187'

   dataset:
     dataset_name: 'Hand_gesture_dataset'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_04_00/datasets/hand_gesture_dataset.zip'

   data_processing_feature_extraction:
     feature_extraction_name: 'Input256_RAW_256Feature_1Frame_3InputChannel_removeDC_2D1'
     variables: 3

   training:
     model_name: 'CLS_55k_NPU'
     training_epochs: 40
     batch_size: 30
     quantization: 2
     quantization_method: 'QAT'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

   testing:
     enable: True

   compilation:
     enable: True

Dataset Description
-------------------

* **Sensor**: 3-axis accelerometer (X, Y, Z) from TI Sensor BoosterPack
* **Variables**: X, Y, Z acceleration (``variables: 3``)
* **Frame size**: 256 samples per frame
* **Stride**: 0.25 (64-sample step between frames)
* **Normalization**: Range normalization per frame

Feature Extraction
------------------

The example uses raw time-domain features (no FFT). Range normalization captures
gesture amplitude and shape directly.

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Parameters
     - Description
   * - ``CLS_55k_NPU``
     - ~55,000
     - Default — CNN optimized for NPU, best accuracy

Expected Results
----------------

.. code-block:: text

   Accuracy: ~94.46% on test data
   Device: MSPM0G5187 NPU

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller with integrated NPU
* `TI Sensor BoosterPack <https://www.ti.com/tool/BOOSTXL-SENSORS>`_ with 3-axis accelerometer

**Software**

* Code Composer Studio 12.x or later
* MSPM0 SDK 2.10.04 or later

Next Steps
----------

* Learn about feature extraction: :doc:`/features/feature_extraction`
* Deploy to device: :doc:`/deployment/npu_device_deployment`
* Browse similar examples: :doc:`fall_detection_classification`, :doc:`har_activity_recognition`
