===============
Cough Detection
===============

Binary audio classification (cough versus other) running fully on the LP-MSPM0G5187 LaunchPad, using LPC
spectral envelope features and a TCDS-ResNet on the NPU. New in 1.5.0.

Example folder: ``examples/cough_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/cough_detection>`_)

Overview
--------

* **Task**: audio classification (``audio_classification``, ``target_module: audio``), 2 classes (cough, other)
* **Application**: health monitoring in hospital or wearable settings, where a missed cough is worse than a false alarm
* **Device**: MSPM0G5187 (Arm Cortex-M0+ at 80 MHz, TinyEngine NPU, 128 KB flash, 32 KB SRAM). The folder has a single
  ``config.yaml``.
* **Model**: ``TCDS_ResNet_NPU``
* **Features**: preset ``MSPM0_LPC_8820`` (linear predictive coding, matched to the firmware)

Data Collection
---------------

Audio was captured with TI EdgeAI Studio connected to the LaunchPad over USB. A PCM1809 MEMS microphone feeds the
I2S peripheral at 44.1 kHz; the firmware keeps the left channel, converts samples to 16 bits and decimates 5:1 to
8820 Hz mono. EdgeAI Studio stores the raw FIFO samples as a labeled CSV file per recording. The recording set is
about 6 minutes in total: three cough recordings (two male, one female) and three speech recordings (greetings,
meeting speech, chatting), about 57 seconds each. Silence and white noise were recorded only as a noise bank for
augmentation.

Preparing the Dataset
---------------------

The config downloads the prepared dataset from
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/cough_detection.zip``. To prepare your own recordings, use
``prepare_dataset.py``:

.. code-block:: bash

   cd examples/cough_detection
   python prepare_dataset.py --data-dir /path/to/csvs --output-dir /path/to/output

The script mirrors the firmware signal path so that training clips equal what the firmware produces:

#. Extract int16 samples (shift right by 10, clamp).
#. Deinterleave and keep the left channel (44.1 kHz).
#. Decimate by 5 to 8820 Hz with plain integer decimation and no anti-alias filter. This must match the firmware,
   otherwise training and inference see different aliasing.
#. Cut into 2-second clips (17,640 samples) with 50% overlap.
#. Augment the cough class with gain 1.0 (original) and gain 0.15 plus a 10 dB SNR noise mix.

A cough is naturally louder than speech. Without the attenuated copies the model would learn to threshold on
loudness. The attenuated copies make both classes span the same RMS range, so the model must discriminate on
spectral shape. The v2 dataset in the readme has 330 cough clips (165 original, 165 augmented) and 165 other clips.

Feature Extraction: MSPM0_LPC_8820
----------------------------------

Every parameter of this preset matches a constant in the firmware (``lpc_defs.h``, ``feature_extraction.c``). If the
training and firmware parameters differ, accuracy collapses.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Value
     - Note
   * - ``sampling_rate``
     - 8820 Hz
     - 44100 / 5
   * - ``audio_duration_ms``
     - 2000 ms
     - 100 frames of 20 ms
   * - ``audio_feature``
     - LPC
     - Linear predictive coding
   * - ``lpc_order``
     - 10
     - Order of the all-pole vocal tract model
   * - ``nlpc``
     - 70
     - Frequency bins of the envelope
   * - ``frame_length_ms`` / ``frame_step_ms``
     - 30 ms / 20 ms
     - 264-sample window, 176-sample step

Per 20 ms frame the preset applies pre-emphasis, a Q15 Hamming window, autocorrelation (lags 0 to 10), a
Levinson-Durbin recursion, evaluates the frequency response at 70 log-spaced bins (112 Hz to 3946 Hz), converts it
to dB and quantizes it to INT8. The model input is ``(1, 1, 100, 70)`` INT8. Cough gives a broadband, flat spectral
envelope, speech gives narrow formant peaks between 700 and 3000 Hz, which LPC captures in only 70 numbers.

Model
-----

``TCDS_ResNet_NPU`` (Temporal Channel-Decoupled Separable ResNet) is designed for INT8-only execution, a small
activation memory and 1D temporal convolutions. The input is permuted from ``(N, 1, T, F)`` to ``(N, F, T, 1)`` so
the 70 LPC bins become channels and every kernel is ``K x 1``.

.. list-table::
   :header-rows: 1
   :widths: 20 30 15 15 20

   * - Stage
     - Block
     - In ch.
     - Out ch.
     - Stride
   * - Stem
     - Conv2d 3x1
     - 70
     - 16
     - 1
   * - Blocks 1 and 2
     - TCDSBasicBlock 9x1
     - 16, 24
     - 24, 24
     - 2, 1
   * - Blocks 3 and 4
     - TCDSBasicBlock 9x1
     - 24, 32
     - 32, 32
     - 2, 1
   * - Blocks 5 and 6
     - TCDSBasicBlock 9x1
     - 32, 48
     - 48, 48
     - 2, 1
   * - Head
     - GAP, Dropout 0.5, FC
     - 48
     - 2 classes
     -

Each block is a depthwise ``K x 1`` convolution plus a pointwise 1x1 convolution with BatchNorm and ReLU. The
blocks have no residual connection, which avoids the quantized-add incompatibility with TI QAT. The readme
quotes a depthwise-separable factorization saving of about 8 to 10 times in parameters. The audio models enabled
in 1.5.0 are ``DSCNN_NPU``, ``DSCNN_32K_NPU`` (a compact DSCNN for LPC features), ``DSCNN_GB_NPU``,
``TCDS_ResNet_NPU`` and ``TCDS_ResNet_FB_NPU``.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/cough_detection/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\cough_detection\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: audio
     task_type: audio_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: cough_detection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/cough_detection.zip
   data_processing_feature_extraction:
     feature_extraction_name: MSPM0_LPC_8820
   training:
     model_name: TCDS_ResNet_NPU
     batch_size: 64
     training_epochs: 40
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 1
     learning_rate: 0.01
     weight_decay: 4e-4
   testing: {}
   compilation:
     compile_preset_name: default_preset

``num_gpus: 1`` expects a CUDA GPU; use 0 for CPU. The learning rate is kept low for the small dataset. The
compilation step runs the TI Neural Network Compiler and produces ``model.a`` and ``tvmgen_default.h`` for the
MSPM0G5187 NPU. For the audio task type see :doc:`/task_types/audio_classification`.

Results
-------

The example readme reports these results on its v2 dataset (562 clips, 80/20 train and validation split):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Metric
     - Value
   * - Best validation accuracy
     - 88.5%
   * - Cough recall
     - 100% (0 false negatives)
   * - Cough precision / F1
     - about 85% / about 92%

The validation confusion matrix has 99 correct coughs, 0 missed coughs, 17 false positives and 32 correct others.
The false positives are speech segments that sound like a cough. The high recall is intentional because missing a
cough is the worse error; the firmware debounce (``DETECT_THRESH = 8`` frames, 160 ms of consecutive cough
predictions) filters out isolated false positives.

The automated 1.5.0 config test run (early stopped, test set of 50 clips) reached a test accuracy of 78.00% and an
AUC ROC of 0.979, so it is below the readme figures. Expect the result to vary with the data and training length.

Memory Footprint
----------------

Reported by the TI Neural Network Compiler for MSPM0G5187 in the readme:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Segment
     - Size
     - Memory
   * - Code (firmware glue)
     - 4.58 KB
     - Flash
   * - Read-only data (weights and parameters)
     - 45.04 KB
     - Flash
   * - Read-write data (activations)
     - 14.45 KB
     - SRAM

That is 49.62 KB of 128 KB flash (39%) and 14.45 KB of 32 KB SRAM (45%).

Output Location
---------------

Results are written to ``./data/projects/cough_detection/run/<date-time>/TCDS_ResNet_NPU/``. See
:doc:`/getting_started/understanding_run_output` and
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme` for deployment.

See Also
--------

* :doc:`wake_word_detection`
* :doc:`glass_break_detection`
* :doc:`google_speech_command`
