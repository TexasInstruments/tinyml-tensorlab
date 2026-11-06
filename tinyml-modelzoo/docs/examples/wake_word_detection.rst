===================
Wake Word Detection
===================

Always-on detection of the wake word "OK Kilby" on the MSPM0G5187 with NPU, using a learned filterbank front-end
and a TCDS-ResNet classifier. New in 1.5.0.

Example folder: ``examples/wake_word_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/wake_word_detection>`_)

Overview
--------

* **Task**: audio classification (``audio_classification``, ``target_module: audio``), 2 classes
* **Application**: low-power, privacy-preserving voice interfaces for smart home, industrial and wearable devices.
  The wake word is detected on the device so no audio leaves it, and downstream voice-command processing is
  triggered only after a detection.
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config in the folder)
* **Model**: ``TCDS_ResNet_FB_NPU``
* **Targets in the readme**: under 200 ms detection latency from the end of the utterance, a high detection rate
  with a low false-accept rate, and low average power for always-on, battery-operated listening

The Dataset
-----------

The wake word is "OK Kilby". The data is a TI-internal dataset collected for this application and is not derived
from a public dataset. It contains multiple speakers saying the wake word with varying tone, distance and speaking
rate, background noise (household, office, ambient) and non-wake-word speech to reduce false accepts.

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Class
     - Files in this example
   * - Wake word ("OK Kilby")
     - 6000
   * - Non-wake-word / background
     - 4600

The subset used by the example is downloaded by the config from
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wake_word_detection.zip``. The readme notes that it is a
small part of an internal dataset of more than 50 GB, so for a production-quality model you should collect and
add your own wake word and background data (speakers, accents, distances, microphones and noise conditions).

Feature Extraction: Learned Filterbank
--------------------------------------

Instead of a fixed MFCC or FFT front-end, the model learns its filterbank jointly with the classifier. The config
uses the preset ``WakeWordDetection_Filterbank_Default``.

#. Audio is sampled at 8 kHz with a 2-second window per inference (16,000 samples).
#. A strided Conv1D filterbank (kernel 64, stride 4) slides over the raw audio and projects it into 32 channels (``fb_output_channel`` of the ``WakeWordDetection_Filterbank_Default`` preset).
#. A max-pool over each 20 ms context (160 samples) produces one feature vector per frame.
#. Filterbank weights are quantized to 2 bits and the input audio to 16 bits.
#. The features feed directly into the TCDS-ResNet backbone.

The commented lines in the config show the underlying preset keys (``audio_feature: "FB"``, ``fb_conv_kernel``,
``fb_output_channel``, ``fb_bitwidth``, ``fb_context_ms``, ``fb_conv_stride``, ``input_bit_depth``).

Model
-----

``TCDS_ResNet_FB_NPU`` combines the learned filterbank with a Temporal Channel-Decoupled Separable ResNet:

* Stem: Conv3x1, BatchNorm, ReLU (32 channels)
* Backbone: 3 TCDS basic blocks (channels 32, 48, 64, 96). Each block has a main branch of depthwise 9x1,
  pointwise 1x1, depthwise 9x1, pointwise 1x1, and a skip branch (1x1 projection when the stride or channel count
  changes). The output is ReLU(main + skip).
* Head: Dropout, Flatten, Fully Connected, 2 outputs
* Mixed precision: stem and head at 8 bits, residual blocks at 2 bits

The non-filterbank variant ``TCDS_ResNet_NPU`` is also available. The audio models enabled in 1.5.0 are
``DSCNN_NPU``, ``DSCNN_32K_NPU``, ``DSCNN_GB_NPU``, ``TCDS_ResNet_NPU`` and ``TCDS_ResNet_FB_NPU``.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/wake_word_detection/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\wake_word_detection\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: audio
     task_type: audio_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: wake_word_detection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wake_word_detection.zip
   data_processing_feature_extraction:
     feature_extraction_name: WakeWordDetection_Filterbank_Default
     sampling_rate: 8000
     audio_duration_ms: 2000
   training:
     model_name: TCDS_ResNet_FB_NPU
     batch_size: 4096
     training_epochs: 100
     num_gpus: 1
     quantization: 2
     learning_rate: 0.01
     weight_decay: 1e-5
     auto_quantization: False
   testing: {}
   compilation: {}

``quantization: 2`` with ``auto_quantization: False`` runs quantization-aware training with the mixed-precision
scheme above. The example uses a batch size of 4096 and a learning rate of 0.01, as in the config shown here.
``num_gpus: 1`` expects a GPU; use 0 for CPU. For the audio
task type see :doc:`/task_types/audio_classification`.

Results
-------

The example readme does not publish accuracy numbers. In the automated 1.5.0 config test run (test set of 1080
samples: 600 wake word, 480 others) the quantized model reached a test accuracy of 96.20% and an AUC ROC of
0.991. Treat these as indicative results from a short test run on the small subset.

Output Location
---------------

Results are written to ``./data/projects/wake_word_detection/run/<date-time>/TCDS_ResNet_FB_NPU/``. The compiled
model and interface headers are in ``compilation/artifacts/``. See
:doc:`/getting_started/understanding_run_output` and
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme`.

Running on Device
-----------------

Build the deployment with the MSPM0 SDK (2.08.00 or later) and Code Composer Studio, with a microphone as audio
input. The compiler flow uses the TI Neural Network Compiler; see :doc:`/devices/index` for MSPM0G5187.

See Also
--------

* :doc:`google_speech_command`
* :doc:`glass_break_detection`
* :doc:`cough_detection`
