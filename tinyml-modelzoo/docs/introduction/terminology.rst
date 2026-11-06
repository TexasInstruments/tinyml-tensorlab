===========
Terminology
===========

Key terms and abbreviations used in the Tiny ML ModelZoo documentation.

General ML Terms
----------------

.. glossary::
   :sorted:

   Quantization
      Reducing the precision of model weights and activations from 32-bit floating
      point to lower bit-widths. This reduces model size and speeds up inference on MCUs.

   QAT
      **Quantization-Aware Training**. Quantization effects are simulated during
      training, which usually gives better accuracy than post-training quantization.

   PTQ
      **Post-Training Quantization**. Quantization applied after training. Faster than
      QAT but may lose more accuracy.

   Mixed-precision quantization
      Using different bit-widths for different layers, chosen automatically (Hessian-aware, HAWQ).

   Autoencoder
      A network that compresses input data and reconstructs it. Used for anomaly detection.

   Feature Extraction
      Transforming raw input data (for example time-series signals) into a representation
      for the neural network. Examples: FFT, wavelets, binning, MFCC.

   Early stopping
      Ending training when validation accuracy has not improved for a number of epochs
      (``early_stopping_patience``).

   Inference
      Using a trained model to make predictions on new data.

   MACs
      **Multiply-Accumulate Operations**. A measure of computational complexity.

   ONNX
      **Open Neural Network Exchange**. The model format used between training and compilation.

   Golden vectors
      Real input samples with the outputs the model should produce, written to
      ``test_vector.c`` so a device implementation can be checked against the PC.

Tiny ML ModelZoo Terms
----------------------

.. glossary::
   :sorted:

   ModelZoo
      The ``tinyml-modelzoo`` repository and the ``tinyml_modelzoo`` package: model
      definitions, example configurations and run scripts. The entry point for users.

   ModelMaker
      The ``tinyml_modelmaker`` package that orchestrates training and compilation.

   TinyVerse
      The ``tinyml_tinyverse`` package with training scripts, dataset loaders and utilities.

   Tensorlab
      ``tinyml-tensorlab``, the name of the toolchain up to release 1.4.0. It is deprecated
      as the entry point and replaced by ``tinyml-modelzoo``.

   TINPU
      **TI Neural Processing Unit** (TinyEngine NPU). Hardware accelerator for neural
      network inference in select TI MCUs (F28P55, MSPM0G5187, AM13E2).

   NPU
      **Neural Processing Unit**. General term; TINPU is TI's implementation.

   NNC
      **Neural Network Compiler** (``ti_mcu_nnc``). TI's compiler that turns ONNX models
      into optimized code for TI MCUs.

   NAS
      **Neural Architecture Search**. Automated search for network architectures.

   BYOD
      **Bring Your Own Data**. Using your own dataset instead of a built-in example dataset.

   BYOM
      **Bring Your Own Model**. Adding your own model architecture to the model zoo or
      compiling an external ONNX model.

Device and Hardware Terms
-------------------------

.. glossary::
   :sorted:

   C2000
      TI's family of 32-bit real-time MCUs for control applications (C28 DSP: F28P55,
      F28P65, F2837, ... and C29: F29H85, F29P58, F29P32).

   MSPM0
      TI's ultra-low-power Arm Cortex-M0+ MCUs.

   MSPM33C
      TI's Arm Cortex-M33 MCUs.

   AM13
      TI's Arm Cortex-M33 MCUs. AM13E2 has an NPU.

   AM26x
      TI's Arm Cortex-R5 MCUs (AM263, AM263P, AM261).

   CCS
      **Code Composer Studio**, TI's IDE for MCU development.

   C2000Ware
      TI's software development kit for C2000 MCUs.

   LaunchPad
      TI's low-cost development kit for evaluating MCUs.

Configuration Terms
-------------------

.. glossary::
   :sorted:

   task_type
      The ML task, for example ``generic_timeseries_classification``,
      ``generic_timeseries_regression``, ``generic_timeseries_forecasting``,
      ``generic_timeseries_anomalydetection``, ``arc_fault``, ``audio_classification``,
      ``image_classification``, ``radar_point_cloud_classification`` or ``radar_surface_classification``.

   target_device
      The deployment MCU, for example ``F28P55``, ``MSPM0G3507`` or ``AM263``.

   feature_extraction_name
      A preset feature extraction pipeline, for example ``Generic_1024Input_FFTBIN_64Feature_8Frame``.

   model_name
      The model architecture, for example ``CLS_1k_NPU``.

   projects_path
      ``common.projects_path``: where run output is written (default ``./data/projects``).

   frame_size
      The number of samples in each data frame (window).

   stride_size
      The step between consecutive frames, as a fraction of the frame size.

   variables
      The number of input channels in the dataset.

Model Size Conventions
----------------------

Model names often include a size indicator, for example ``CLS_1k_NPU`` or ``CLS_4k_NPU``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Name part
     - Meaning
   * - ``_100``, ``_1k``, ``_4k``, ``_13k``
     - Approximate parameter count (100, 1,000, 4,000, 13,000)
   * - ``_NPU``
     - Architecture follows the NPU constraints

See :doc:`/appendix/model_zoo_reference` for the exact models in 1.5.0.

Abbreviations
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Abbreviation
     - Full form
   * - CNN
     - Convolutional Neural Network
   * - DSP
     - Digital Signal Processor
   * - FFT
     - Fast Fourier Transform
   * - LSTM
     - Long Short-Term Memory
   * - MCU
     - Microcontroller Unit
   * - MFCC
     - Mel-Frequency Cepstral Coefficients
   * - ODL
     - On-Device Learning
   * - PMSM
     - Permanent Magnet Synchronous Motor
   * - SMAPE
     - Symmetric Mean Absolute Percentage Error
   * - SRAM
     - Static Random Access Memory
