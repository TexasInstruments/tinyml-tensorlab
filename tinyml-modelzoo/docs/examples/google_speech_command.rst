=================================
Google Speech Command Recognition
=================================

12-class keyword spotting from 1-second audio clips, using MFCC features and a depthwise separable CNN (DSCNN)
on the MSPM0G5187 with NPU.

Example folder: ``examples/google_speech_command``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/google_speech_command>`_)

Overview
--------

* **Task**: audio classification (``audio_classification``, ``target_module: audio``), 12-class keyword spotting
* **Application**: voice command recognition
* **Dataset**: Google Speech Commands v0.02, 12-class variant
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config in the folder)
* **Model**: ``DSCNN_NPU``

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
     - ``_unknown_``: all other words (for example bird, cat, tree)
   * - Silence
     - ``_silence_``: 1-second clips cut from ``_background_noise_``

The Dataset
-----------

The config downloads the prepared dataset from
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/google_speech_commands_12class.zip``, so you can run the
example directly. To regenerate it from the original data, run the helper script (needs ``torch``,
``torchaudio``, ``scipy``, ``pydub``, ``numpy`` and ``tqdm``) and point ``dataset.input_data_path`` at the
result:

.. code-block:: bash

   cd examples/google_speech_command
   python generate_dataset.py

The script downloads Google Speech Commands v0.02 through TorchAudio, copies the 10 keyword classes into their own
folders, maps all other words into ``_unknown_`` (prefixing the original label to avoid file name collisions), and
cuts the background noise files into 1-second windows with 50% overlap for ``_silence_``. The output is
``SpeechCommands/classes/<class>/``.

Features
--------

The config uses the feature extraction preset ``GoogleSpeechCommands_MFCC_Default``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Value
   * - Sampling rate
     - 16000 Hz
   * - Audio duration
     - 1000 ms
   * - Frame length / step
     - 30 ms / 20 ms
   * - MFCC coefficients
     - 10
   * - Mel bins
     - 40

The output feature shape is ``[N, 1, 49, 10]``: batch, 1 channel, 49 time frames and 10 MFCC coefficients.

Model
-----

``DSCNN_NPU`` is a depthwise separable CNN designed for efficient inference on the NPU: a 10x4 convolution with
stride 2, dropout, four blocks of depthwise 3x3 plus pointwise 1x1 convolution, dropout, adaptive average pooling
and a fully connected layer to 12 classes (64 filters). Splitting a convolution into depthwise (spatial) and
pointwise (channel mixing) parts cuts computation and model size while keeping good keyword spotting accuracy.
Other audio models enabled in 1.5.0 are ``DSCNN_32K_NPU``, ``DSCNN_GB_NPU``, ``TCDS_ResNet_NPU`` and
``TCDS_ResNet_FB_NPU`` (see :doc:`wake_word_detection`).

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/google_speech_command/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\google_speech_command\config_MSPM0.yaml

Configuration
-------------

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

``num_gpus: 1`` expects a GPU; use 0 for CPU. Training on this dataset is long compared with the other examples.
See :doc:`/task_types/audio_classification` for the audio task type.

Results
-------

The example readme does not publish accuracy numbers. In the automated 1.5.0 config test run the training
reached a best quantized epoch (epoch 7) with a validation accuracy of 91.30%, an F1 score of 0.913 and an AUC
ROC of 0.997. That run hit the test harness time limit before it finished, so it is a partial result and not a
converged 20-epoch result.

Output Location
---------------

Results are written to ``./data/projects/google_speech_commands_12class/run/<date-time>/DSCNN_NPU/``. See
:doc:`/getting_started/understanding_run_output` for the generated files and
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme` for device deployment. The MSPM0 SDK
2.10.04 or later was listed for this example in 1.4.0.

See Also
--------

* :doc:`wake_word_detection`
* :doc:`glass_break_detection`
* :doc:`cough_detection`
