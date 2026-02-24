===========
Terminology
===========

This glossary defines key terms and abbreviations used throughout the
Tiny ML Tensorlab documentation.

General ML Terms
----------------

.. glossary::
   :sorted:

   Quantization
      The process of reducing the precision of model weights and activations
      from floating-point (32-bit) to lower bit-widths (2/4/8-bit). This reduces
      model size and improves inference speed on MCUs.

   QAT
      **Quantization-Aware Training**. A technique where quantization effects
      are simulated during training, resulting in better accuracy compared to
      post-training quantization.

   PTQ
      **Post-Training Quantization**. Quantization applied after training is
      complete. Faster than QAT but may result in lower accuracy.

   Autoencoder
      A neural network architecture that learns to compress input data into
      a lower-dimensional representation (encoding) and then reconstruct it
      (decoding). Used for anomaly detection in Tiny ML Tensorlab.

   Feature Extraction
      The process of transforming raw input data (e.g., time-series signals)
      into a more meaningful representation for the neural network. Examples
      include FFT, wavelets, and binning.

   Inference
      The process of using a trained model to make predictions on new data.

   MACs
      **Multiply-Accumulate Operations**. A measure of computational complexity.
      One MAC = one multiplication followed by one addition.

   ONNX
      **Open Neural Network Exchange**. A standard format for representing
      machine learning models, used as the intermediate format before compilation.

Tiny ML Tensorlab Terms
-----------------------

.. glossary::
   :sorted:

   ModelZoo
      The ``tinyml-modelzoo`` repository containing pre-defined model architectures
      and example configurations. The primary entry point for users.

   ModelMaker
      The ``tinyml-modelmaker`` repository that orchestrates the training and
      compilation workflow.

   TinyVerse
      The ``tinyml-tinyverse`` repository containing core training scripts,
      dataset loaders, and utilities.

   TINPU
      **TI Neural Processing Unit**. Hardware accelerator for neural network
      inference present in select TI MCUs (F28P55, AM13E2, MSPM0G5187).

   NPU
      **Neural Processing Unit**. General term for hardware acceleration of
      neural network operations. TINPU is TI's implementation.

   NNC
      **Neural Network Compiler**. TI's compiler that converts ONNX models
      into optimized code for TI MCUs.

   NAS
      **Neural Architecture Search**. An automated technique for discovering
      optimal neural network architectures. Available in Tiny ML Tensorlab for
      time series classification.

   GoF Test
      **Goodness of Fit Test**. A visualization tool in Tiny ML Tensorlab that
      helps evaluate whether a dataset is suitable for classification by
      plotting class separability using PCA and t-SNE.

Device & Hardware Terms
-----------------------

.. glossary::
   :sorted:

   C2000
      A family of TI 32-bit real-time microcontrollers optimized for
      control applications. Includes devices like F28P55, F28P65, F2837.

   MSPM0
      A family of TI ultra-low-power Arm Cortex-M0+ microcontrollers.

   MSPM33C
      A family of TI Arm Cortex-M33 microcontrollers with TrustZone security.

   AM13
      A family of TI Arm Cortex-M33 microcontrollers with NPU acceleration.
      Includes AM13E2.

   CCS
      **Code Composer Studio**. TI's integrated development environment (IDE)
      for programming TI microcontrollers.

   C2000Ware
      TI's software development kit (SDK) for C2000 microcontrollers.

   LaunchPad
      TI's low-cost development kit for evaluating MCUs.

Configuration Terms
-------------------

.. glossary::
   :sorted:

   task_type
      Configuration parameter specifying the ML task. Options include:
      ``generic_timeseries_classification``, ``generic_timeseries_regression``,
      ``generic_timeseries_forecasting``, ``generic_timeseries_anomalydetection``,
      ``image_classification``.

   target_device
      Configuration parameter specifying the deployment MCU. Examples:
      ``F28P55``, ``MSPM0G3507``, ``AM263``.

   feature_extraction_name
      Configuration parameter specifying a preset feature extraction pipeline.
      Example: ``Generic_1024Input_FFTBIN_64Feature_8Frame``.

   model_name
      Configuration parameter specifying which model architecture to use.
      Example: ``CLS_1k_NPU``, ``REGR_500_NPU``.

Data Terms
----------

.. glossary::
   :sorted:

   BYOD
      **Bring Your Own Data**. The practice of using your own dataset with
      Tiny ML Tensorlab rather than a built-in example dataset.

   BYOM
      **Bring Your Own Model**. The practice of adding custom model architectures
      to the ModelZoo or compiling external ONNX models.

   frame_size
      The number of samples in each data frame (window) used for training.

   stride_size
      The overlap between consecutive frames, expressed as a fraction.
      ``stride_size: 0.5`` means 50% overlap.

   variables
      The number of input channels or features in the dataset.
      For multi-axis sensor data, this equals the number of axes.

Model Size Conventions
----------------------

Model names in Tiny ML Tensorlab often include size indicators:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Suffix
     - Meaning
     - Parameter Count
   * - ``_100``
     - ~100 parameters
     - Ultra-minimal
   * - ``_1k`` or ``_1K``
     - ~1,000 parameters
     - Small
   * - ``_4k`` or ``_4K``
     - ~4,000 parameters
     - Medium
   * - ``_13k`` or ``_13K``
     - ~13,000 parameters
     - Large
   * - ``_NPU``
     - NPU-optimized
     - Architecture follows NPU constraints

Abbreviations
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Abbreviation
     - Full Form
   * - AI
     - Artificial Intelligence
   * - CNN
     - Convolutional Neural Network
   * - DSP
     - Digital Signal Processor
   * - FFT
     - Fast Fourier Transform
   * - GPU
     - Graphics Processing Unit
   * - LSTM
     - Long Short-Term Memory (neural network)
   * - MCU
     - Microcontroller Unit
   * - ML
     - Machine Learning
   * - MSE
     - Mean Squared Error
   * - NN
     - Neural Network
   * - PMSM
     - Permanent Magnet Synchronous Motor
   * - SMAPE
     - Symmetric Mean Absolute Percentage Error
   * - SRAM
     - Static Random Access Memory
