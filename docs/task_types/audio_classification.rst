====================
Audio Classification
====================

Audio classification assigns a label to a short audio clip, for example a spoken keyword, a cough or the
sound of breaking glass. It is new in Tiny ML ModelZoo 1.5.0.

Overview
--------

**What it does**: Takes a fixed-length audio clip, extracts audio features on the device (MFCC, LPC,
filterbank or FFT), and outputs a class label.

**Example**: 1 second of 16 kHz speech -> "yes", "no", "stop", ..., ``_unknown_`` or ``_silence_``

**Use cases** (each has a ready-made example):

* Keyword spotting: :doc:`/examples/google_speech_command`
* Wake word detection: :doc:`/examples/wake_word_detection`
* Cough detection: :doc:`/examples/cough_detection`
* Glass break detection: :doc:`/examples/glass_break_detection`

Task Settings
-------------

.. list-table::
   :widths: 30 70

   * - ``task_type``
     - ``audio_classification``
   * - ``target_module``
     - ``audio`` (inferred from ``task_type`` if you omit it)
   * - Target device
     - ``MSPM0G5187`` (the only audio target device in 1.5.0)
   * - Data folder
     - ``classes/<class_name>/*.wav`` (or ``*.flac``)
   * - Early stopping
     - On by default in 1.5.0 (``training.early_stopping``, ``early_stopping_patience: 5``);
       see :doc:`/features/early_stopping`

Configuration
-------------

This is ``examples/google_speech_command/config_MSPM0.yaml``:

.. code-block:: yaml

   common:
     target_module: audio
     task_type: audio_classification
     target_device: MSPM0G5187

   dataset:
     dataset_name: google_speech_commands_12class
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/google_speech_commands_12class.zip

   data_processing_feature_extraction:
     feature_extraction_name: GoogleSpeechCommands_MFCC_Default
     sampling_rate: 16000
     audio_duration_ms: 1000

   training:
     model_name: DSCNN_NPU
     batch_size: 64
     training_epochs: 20
     num_gpus: 1
     quantization: 2
     learning_rate: 0.1
     weight_decay: 1e-5

   testing: {}
   compilation: {}

Run it from the tinyml-modelzoo folder:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/google_speech_command/config_MSPM0.yaml

See :doc:`/getting_started/quickstart` for install options.

Dataset Format
--------------

.. code-block:: text

   my_audio_dataset/
   └── classes/
       ├── class_a/
       │   ├── clip1.wav
       │   └── clip2.wav
       └── class_b/
           └── clip1.wav

Each folder under ``classes/`` is one class. Annotation lists are optional and are generated from
``split_type`` and ``split_factor`` when missing. Details, plus the scripts that prepare public datasets, are
in :doc:`/byod/audio_radar_format`.

Feature Extraction Presets
--------------------------

Select a preset with ``data_processing_feature_extraction.feature_extraction_name``. Set ``sampling_rate``
and ``audio_duration_ms`` to match your clips.

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Preset
     - Audio feature
     - Used by
   * - ``GoogleSpeechCommands_MFCC_Default``
     - MFCC (10 coefficients, 40 mel bands, 30 ms frames, 20 ms step)
     - Google Speech Commands
   * - ``WakeWordDetection_Filterbank_Default``
     - Learned filterbank (``FB``)
     - Wake word detection
   * - ``MSPM0_LPC_8k``
     - LPC (order 10), 8000 Hz, 2000 ms
     - LPC-based examples
   * - ``MSPM0_LPC_8820``
     - LPC (order 10), 8820 Hz, 2000 ms
     - Cough detection
   * - ``GlassBreakDetection_512FFT_Default``
     - 512-point Q15 FFT with binning (32 features per frame, 86 frames)
     - Glass break detection

Available Models
----------------

The audio models are defined in ``tinyml-modelmaker`` (``audio_classification.py``) and implemented in
``tinyml_modelzoo/models/audio.py``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Model Name
     - Description
   * - ``DSCNN_NPU``
     - Depthwise-separable CNN for audio classification (Google Speech Commands example)
   * - ``DSCNN_32K_NPU``
     - SRAM/flash-optimized DSCNN
   * - ``DSCNN_GB_NPU``
     - Glass-break DSCNN with 32 filters and depthwise-separable convolutions
   * - ``TCDS_ResNet_NPU``
     - Temporal channel-decoupled separable ResNet for LPC features (cough detection)
   * - ``TCDS_ResNet_FB_NPU``
     - Temporal CNN with depthwise-separable residual blocks for filterbank features (wake word)

Metrics
-------

The training log reports the same classification metrics as time series classification: accuracy
(``Acc@1``), F1-score and confusion matrix, for the float model (``FloatTrain``) and the quantized model
(``QuantTrain``). See :doc:`/getting_started/understanding_run_output`.

Tips
----

* Make sure ``sampling_rate`` and ``audio_duration_ms`` match the recordings in the dataset and the
  firmware that will run on the device.
* Mono audio is used; the presets set ``mono: True``.
* Keep classes balanced; add an ``_unknown_`` / ``_silence_`` class for keyword spotting, as the Google
  Speech Commands example does.

See Also
--------

* :doc:`/byod/audio_radar_format` - Audio dataset layout and preparation scripts
* :doc:`timeseries_classification` - Time series classification
* :doc:`radar_classification` - The other new 1.5.0 task type
