================================
Dynamic Hand Gesture Recognition
================================

Real-time dynamic hand gesture classification from 3-axis accelerometer data on the MSPM0G5187 with
integrated NPU.

Example folder: ``examples/dynamic_hand_gesture_recognition``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/dynamic_hand_gesture_recognition>`_)

Overview
--------

Buttons and touchscreens are impractical in hands-free or hygienic environments, rule-based gesture logic is
brittle against natural variation, and cloud recognition adds latency and privacy concerns. This example
classifies gestures directly on the MCU.

* **Task**: multi-class classification (4 classes), ``generic_timeseries_classification``
* **Application**: gesture-based human-machine interface
* **Sensor**: 3-axis accelerometer of the TI `Sensor BoosterPack <https://www.ti.com/tool/BOOSTXL-SENSORS>`_
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config)
* **Model**: ``CLS_55k_NPU`` (about 55K parameters)

Gesture Classes
---------------

1. **Circle**: clockwise or counter-clockwise circular motion
2. **Wave**: side-to-side waving motion
3. **Tap**: short impact gesture
4. **Others**: non-gesture or unrecognized motion

System Components
-----------------

* Hardware: `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ with integrated NPU, TI Sensor BoosterPack
* Software: Code Composer Studio 12.x or later, MSPM0 SDK 2.10.04 or later

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/dynamic_hand_gesture_recognition/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\dynamic_hand_gesture_recognition\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: hand_gesture_dataset
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/hand_gesture_dataset.zip
     split_type: within_files
     gof_test: true
   data_processing_feature_extraction:
     feature_extraction_name: HandGesture256Input_RangeNormalize_1Frame
   training:
     model_name: CLS_55k_NPU
     batch_size: 30
     learning_rate: 0.001
     training_epochs: 40
     num_gpus: 0
     gof_test: true
     auto_quantization: false
   testing:
     enable: true
   compilation:
     enable: true

Feature Extraction
------------------

The preset ``HandGesture256Input_RangeNormalize_1Frame`` uses raw time-domain data (no FFT):

1. Raw 3-axis (X, Y, Z) accelerometer data
2. Windowing: 256 samples per frame
3. Stride 0.25 (frames overlap with a step of 64 samples)
4. Range normalization per frame, which captures the gesture amplitude and shape
5. A single frame is passed to the model

``gof_test: true`` additionally runs the Goodness of Fit analysis of class separability.

Training and Deployment
-----------------------

The example uses batch size 30, learning rate 0.001, the default SGD optimizer and 40 epochs. The model is
quantized to INT8 and compiled with the TI Neural Network Compiler for the NPU. See
:doc:`/DEPLOYING_TO_SDK_EXAMPLE` and :doc:`/deployment/npu_device_deployment`.

Results
-------

The example readme reports an accuracy of about 94.46% on the test data. In the automated 1.5.0 config test run,
``config_MSPM0.yaml`` reached a test accuracy of 94.08% with an AUC ROC of 0.993 (52,434 parameters).

See Also
--------

* :doc:`fall_detection_classification` and :doc:`har_activity_recognition`: other accelerometer examples
