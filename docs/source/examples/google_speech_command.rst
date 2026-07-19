==================================
Google Speech Command Recognition
==================================

12-class keyword spotting from audio using MFCC features and DSCNN model on MSPM0G5187 with NPU.

Overview
--------

* **Task**: Audio Classification (12-class keyword spotting)
* **Application**: Voice command recognition, keyword spotting
* **Dataset**: Google Speech Commands v0.02 (12-class variant)
* **Model**: DSCNN (Depthwise Separable CNN)
* **Device**: MSPM0G5187 (NPU-accelerated)

Keyword Classes
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Labels
   * - Known keywords (10)
     - ``down``, ``go``, ``left``, ``no``, ``off``, ``on``, ``right``, ``stop``, ``up``, ``yes``
   * - Unknown
     - ``_unknown_`` — all non-keyword words
   * - Silence
     - ``_silence_`` — 1-second background noise clips

Device Support
--------------

.. list-table::
   :header-rows: 1
   :widths: 40 50 10

   * - Device
     - Notes
     - Configuration File
   * - ``MSPM0G5187``
     - MSPM0 with NPU
     - ``config_MSPM0.yaml``

Running the Example
-------------------

**Step 1: Generate dataset**

The dataset must be prepared before training:

.. code-block:: bash

   cd tinyml-modelzoo/examples/google_speech_command
   python generate_dataset.py

This downloads Google Speech Commands v0.02 via TorchAudio and prepares a TensorLab-ready structure under ``SpeechCommands/classes/``.

**Step 2: Run training**

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/google_speech_command/config_MSPM0.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\google_speech_command\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'audio_classification'
     target_device: 'MSPM0G5187'

   dataset:
     dataset_name: 'google_speech_commands_12class'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/google_speech_commands_12class.zip'

   data_processing_feature_extraction:
     feature_extraction_name: 'GoogleSpeechCommands_MFCC_Default'

   training:
     model_name: 'DSCNN_NPU'
     training_epochs: 20
     batch_size: 64
     learning_rate: 0.1
     weight_decay: 1e-5
     quantization: 2

   testing:
     enable: True

   compilation:
     enable: True

Feature Extraction (MFCC)
--------------------------

MFCCs (Mel Frequency Cepstral Coefficients) compactly represent speech frequency characteristics for keyword spotting.

.. list-table::
   :header-rows: 1
   :widths: 40 30

   * - Parameter
     - Value
   * - Sampling rate
     - 16000 Hz
   * - Audio duration
     - 1000 ms
   * - Frame length
     - 30 ms
   * - Frame step
     - 20 ms
   * - MFCC coefficients
     - 10
   * - Mel bins
     - 40

Output feature shape: ``[N, 1, 49, 10]`` (batch × 1 channel × 49 time frames × 10 MFCCs)

Model: DSCNN
------------

Depthwise Separable CNN splits standard convolution into depthwise (spatial filtering) + pointwise (channel mixing) operations, reducing computation while maintaining accuracy.

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Filters
     - Description
   * - ``DSCNN_NPU``
     - 64
     - Depthwise separable CNN optimized for NPU, 12-class output

**Architecture**: Conv10×4 / stride 2 → Dropout → (Depthwise3×3 + Pointwise1×1) ×4 → Dropout → AdaptiveAvgPool → FC (12 classes)

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ microcontroller with integrated NPU
* Microphone input

**Software**

* Code Composer Studio (CCS) 12.x or later
* MSPM0 SDK 2.10.04 or later
* Additional Python dependencies: ``torch``, ``torchaudio``, ``scipy``, ``pydub``, ``numpy``

Next Steps
----------

* Learn about audio task type: :doc:`/task_types/index`
* Deploy to device: :doc:`/deployment/npu_device_deployment`
* Browse similar examples: :doc:`dynamic_hand_gesture_recognition`
