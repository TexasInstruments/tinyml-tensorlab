=====================
Glass Break Detection
=====================

Detect glass breaking events from their acoustic signature with FFT features and a compact DSCNN on the
MSPM0G5187 with NPU. New in 1.5.0.

Example folder: ``examples/glass_break_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/glass_break_detection>`_)

Overview
--------

* **Task**: audio classification (``audio_classification``, ``target_module: audio``), 3 classes
* **Application**: security systems, intrusion detection and smart home safety. Threshold-based detectors have
  high false positive rates, while a learned acoustic pattern model is more robust. Processing is on the device,
  so there is no cloud dependency.
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config in the folder)
* **Model**: ``DSCNN_GB_NPU``
* **Targets in the readme**: response time under 100 ms, detection accuracy above 95%, false positive rate
  below 5%, under 32 KB SRAM and 128 KB flash

The Dataset
-----------

The data covers glass breaking (tempered, laminated and plate glass), background noise (household sounds, music,
voices) and similar sounds (dishes breaking, metal impacts, door slams), with 100+ captures per class. Before
augmentation all data comes from public datasets: DEMAND (noise), FSD50K, ESC-50 and Google Speech Commands.

The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/glass_break_detection.zip``. In the 1.5.0 test run the
three classes are ``glass_break``, ``human_speech`` and ``others`` (glass breaking, speech, and all other
sounds).

Feature Extraction
------------------

The config uses the preset ``GlassBreakDetection_512FFT_Default`` on 1-second windows sampled at 44.1 kHz:

#. The window is split into frames of 512 samples.
#. A 512-point real FFT is computed per frame (``FFT_Q15`` in the preset transform chain).
#. The complex magnitude is calculated, the DC component is removed and the Q15 values are scaled.
#. Adjacent bins are averaged (binning) into 32 features per frame.
#. 86 frames are concatenated, giving an input of 86 x 32 features.

The preset transform chain is ``["FFT_Q15", "Q15_SCALE", "Q15_MAG", "BINNING", "CONCAT"]`` with
``frame_size: 512``, ``feature_size_per_frame: 32``, ``num_frame_concat: 86`` and ``q15_scale_factor: 8`` (shown as
comments in the config). See :doc:`/features/feature_extraction`.

Model
-----

``DSCNN_GB_NPU`` is a lightweight depthwise separable CNN. In the 1.5.0 test run the NPU compiler reported code
2.03 KB, read-only data 9.09 KB (flash) and read-write data 15.86 KB (SRAM):

* Input ``(N, 1, 86, 32)``: 86 time frames x 32 FFT features
* BatchNorm, Conv 8x8 stride 4, two depthwise 4x4 plus pointwise 1x1 blocks, Dropout, GlobalAvgPool, FC
* Output: 3 classes
* Mixed precision: depthwise and pointwise convolutions at 4 bits, activations at 8 bits (4-8x smaller than FP32)

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/glass_break_detection/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\glass_break_detection\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: audio
     task_type: audio_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: glass_break_detection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/glass_break_detection.zip
   data_processing_feature_extraction:
     feature_extraction_name: GlassBreakDetection_512FFT_Default
     sampling_rate: 44100
     audio_duration_ms: 1000
   training:
     model_name: DSCNN_GB_NPU
     batch_size: 32
     training_epochs: 20
     num_gpus: 0
     quantization: 2
     learning_rate: 0.001
     weight_decay: 1e-5
     auto_quantization: False
   testing: {}
   compilation: {}

``quantization: 2`` with ``auto_quantization: False`` runs quantization-aware training with the mixed-precision
scheme. For the audio task type see :doc:`/task_types/audio_classification`.

Results
-------

The example readme lists targets (above 95% accuracy) but no measured accuracy. In the automated 1.5.0 config test
run (test set of 1200 samples: 600 glass break, 300 human speech, 300 others) the quantized model reached:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Metric
     - Value
   * - Best quantized epoch
     - 6
   * - Validation accuracy / F1 score (best epoch)
     - 92.72% / 0.927
   * - Test accuracy
     - 92.42%
   * - Test AUC ROC
     - 0.984

This short run did not reach the 95% target. A longer training run, more data, or the augmentation options may
improve it.

Output Location
---------------

Results are written to ``./data/projects/glass_break_detection/run/<date-time>/DSCNN_GB_NPU/``. See
:doc:`/getting_started/understanding_run_output` and
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme` for device deployment with the MSPM0 SDK
(2.08.00 or later) and a microphone input.

See Also
--------

* :doc:`wake_word_detection`
* :doc:`cough_detection`
* :doc:`google_speech_command`
