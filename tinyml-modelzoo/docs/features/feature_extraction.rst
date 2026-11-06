==================
Feature Extraction
==================

Feature extraction transforms raw sensor data into a representation that helps the neural network
learn patterns more effectively. The same transforms run on the PC during training and on the MCU
during inference, so training and deployment see identical features.

* **Reduced input size**: compress long time series.
* **Better patterns**: move to a domain (for example frequency) where patterns are clearer.
* **Faster inference**: smaller inputs mean smaller and faster models.
* **Domain knowledge**: reuse signal processing expertise.

Pipeline
========

Raw data flows through two stages, data processing and then feature extraction:

.. code-block:: text

   Raw signal --> Data processing --> Feature extraction --> Model input
                  (data_proc_transforms)  (feat_ext_transform)
                  e.g. SimpleWindow,      e.g. FFT_FE, BINNING,
                       Downsample              ABS, LOG_DB, CONCAT

.. important::

   ``data_proc_transforms`` run only on the training side (on the PC). ``feat_ext_transform`` runs on
   the PC during training and on the device during inference.

There are two ways to configure it: pick a **preset** with ``feature_extraction_name``, or define a
**custom** pipeline with a ``Custom_*`` name and your own ``feat_ext_transform`` list. The section of
the config YAML is ``data_processing_feature_extraction``.

Configuration parameters
========================

**Core parameters (time series)**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``feature_extraction_name``
     - Preset name (for example ``Generic_1024Input_FFTBIN_64Feature_8Frame``) or a custom name that starts with
       ``Custom_`` (for example ``Custom_Default``). With a ``Custom_*`` name you must give ``feat_ext_transform``.
   * - ``data_proc_transforms``
     - Data processing steps run before feature extraction: ``SimpleWindow``, ``Downsample``, in any order, or ``[]``.
   * - ``feat_ext_transform``
     - Ordered list of feature extraction transforms (see the tables below).
   * - ``variables``
     - Input channels. An integer (first N columns), a list of column indices (``[0, 2, 4]``) or a list of column
       names (``['accel_x', 'accel_y']``).
   * - ``frame_size``
     - Samples per frame.
   * - ``feature_size_per_frame``
     - Output features per frame after the transform.
   * - ``num_frame_concat``
     - Frames concatenated into one model input. Total features = ``feature_size_per_frame x num_frame_concat``.
   * - ``stride_size``
     - Window stride as a fraction of ``frame_size`` (used with ``SimpleWindow``).
   * - ``sampling_rate`` and ``new_sr``
     - Original and target sampling rate for ``Downsample``.
   * - ``scale``
     - Scaling factor applied to the input data.
   * - ``offset``
     - Frame overlap control (see `Frame offset`_).
   * - ``frame_skip``
     - Number of frames skipped between selected frames.
   * - ``stacking``
     - ``'2D1'`` (default) or ``'1D'`` (see `Stacking modes`_).
   * - ``normalize_bin``, ``min_bin``
     - Bin normalization and the first frequency bin to include.
   * - ``analysis_bandwidth``
     - Fraction of the signal bandwidth to analyse (1 = full).
   * - ``log_mul``, ``log_base``, ``log_threshold``
     - Parameters of ``LOG_DB`` (for example 20, 10, 1e-100).
   * - ``q15_scale_factor``
     - Bit-shift scale for the Q15 transforms.
   * - ``gain_variations``
     - Per-class gain augmentation (see `Gain variation augmentation`_).
   * - ``gof_test``
     - Run the Goodness of Fit test on the extracted features; see :doc:`/features/goodness_of_fit`.
   * - ``store_feat_ext_data``
     - Store the extracted features to disk.
   * - ``dont_train_just_feat_ext``
     - Stop after data processing and feature extraction (see `Feature extraction only mode`_).
   * - ``nn_for_feature_extraction``
     - Use a neural network for feature extraction.
   * - ``forecast_horizon``, ``target_variables``
     - Forecasting only: number of future steps and the columns to forecast.

Presets
=======

Presets in 1.5.0 are defined in ModelMaker (``tinyml_modelmaker/ai_modules/timeseries/constants.py``).
Use a preset by name and set ``variables``:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'
     variables: 1

The generic presets follow the pattern ``Generic_<InputSize>Input_<Transform>_<Features>Feature_<Frames>Frame``. For
example, ``Generic_1024Input_FFTBIN_64Feature_8Frame`` takes 1024 samples, applies an FFT with binning to 64
features per frame, and concatenates 8 frames (512 features to the model).

**Generic presets**

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Preset
     - Transform
   * - ``Generic_1024Input_FFTBIN_64Feature_8Frame``, ``Generic_512Input_FFTBIN_32Feature_8Frame``,
       ``Generic_256Input_FFTBIN_16Feature_8Frame``, ``Generic_32Input_FFTBIN_4Feature_8Frame``
     - FFT with binning, 8 frames
   * - ``Generic_1024Input_FFT_512Feature_1Frame``, ``Generic_512Input_FFT_256Feature_1Frame``,
       ``Generic_256Input_FFT_128Feature_1Frame``
     - FFT, single frame
   * - ``Generic_512Input_RAW_512Feature_1Frame``, ``Generic_256Input_RAW_256Feature_1Frame``,
       ``Generic_128Input_RAW_128Feature_1Frame``
     - Raw time-domain samples
   * - ``Generic_8Input_ABS_8Feature_1Frame``
     - Absolute value of 8 samples

**Application presets**

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Preset
     - Application
   * - ``FFT1024Input_256Feature_1Frame_Full_Bandwidth``, ``FFT1024Input_256Feature_1Frame_Half_Bandwidth``,
       ``FFT1024Input_64Feature_4Frame_Half_Bandwidth``, ``FFT1024Input_32Feature_8Frame_Quarter_Bandwidth``,
       ``FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth``
     - FFT with full, half or quarter bandwidth
   * - ``ArcFault_1024Input_256Feature_1Frame_Full_Bandwidth`` (also ``_Half_Bandwidth``, ``ArcFault_1024Input_64Feature_4Frame_Half_Bandwidth``,
       ``ArcFault_1024Input_32Feature_8Frame_Quarter_Bandwidth``,
       ``ArcFault_1024Input_FE_RFFT_128Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth``,
       ``ArcFault_512Input_FE_RFFT_32Feature_8Frame_1InputChannel_removeDC_Full_Bandwidth``)
     - Arc fault
   * - ``Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_1D`` and ``_2D1``,
       ``Input256_FFT_128Feature_1Frame_3InputChannel_removeDC_2D1``,
       ``Input128_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1``,
       ``MotorFault_256Input_FE_RFFT_16Feature_8Frame_3InputChannel_removeDC_1D`` and ``_2D1``,
       ``MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1``
     - Motor fault, 3-axis vibration
   * - ``PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D``, ``PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D_FixedPoint``
     - PIR detection
   * - ``FallDetection_256Input_FE_RFFT_8Feature_8Frame_3InputChannel_removeDC_2D1``
     - Fall detection
   * - ``WiFiCSI_128Input_L2Norm2DFFT_64x26Feature_1Frame_52Subcarriers``
     - WiFi CSI presence detection
   * - ``ECG2500Input_Roundoff_1Frame``
     - ECG classification
   * - ``Custom_ArcFault``, ``Custom_ArcFault_MSPM0``,``Custom_MotorFault``, ``Custom_MotorFault_MSPM0``,
       ``Custom_Default``, ``Custom_Default_MSPM0``
     - Starting points for custom pipelines (``_MSPM0`` variants use the Q15 fixed-point transforms)

The preset list is the one in the source; other presets exist per task type (audio presets are described
below). To see the preset an example uses, open its config in ``examples/<name>/``.

Data processing transforms
==========================

**SimpleWindow** segments continuous data into fixed-size windows:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 256
     stride_size: 0.01
     variables: 1

**Downsample** reduces the sampling rate (``sampling_rate`` to ``new_sr``). Transforms run in the order given:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: ['Downsample', 'SimpleWindow']
     sampling_rate: 313000
     new_sr: 3130
     frame_size: 256
     stride_size: 0.01
     variables: 1

Feature extraction transforms
=============================

``feat_ext_transform`` is an ordered list. Each step processes the output of the previous one.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Transform
     - Description
   * - ``FFT_FE``
     - FFT of each frame.
   * - ``FFT_POS_HALF``
     - Keep the positive-frequency half of the FFT.
   * - ``WINDOWING``
     - Apply a window function.
   * - ``BINNING``
     - Group frequency bins to reduce the feature count.
   * - ``NORMALIZE``, ``RANGE_NORMALIZE``
     - Normalize features (``RANGE_NORMALIZE`` uses the value range).
   * - ``ABS``
     - Absolute value.
   * - ``LOG_DB``
     - Logarithmic (dB) scale.
   * - ``DC_REMOVE``
     - Remove the DC component.
   * - ``CONCAT``
     - Concatenate frames into the final feature vector.
   * - ``RAW_FE``
     - Mean of the wave in each frame; removes DC if ``dc_remove`` is true.
   * - ``ROUND_OFF``
     - Round the values (used by the ``ECG2500Input_Roundoff_1Frame`` ECG preset).
   * - ``FFT_COL``
     - Column-wise real FFT across the variables axis, per window.
   * - ``L2_NORM_ROW``
     - L2 norm across rows.
   * - ``KURT_FE``, ``ENT_FE``, ``ZCR_FE``, ``DOM_FE``, ``SLOPE_FE``
     - Statistical features per frame: kurtosis, spectral entropy, zero-crossing rate, dominant frequency, slope changes.
   * - ``PIR_FE``, ``PIR_FE_Q15``
     - PIR sensor features (float and fixed-point).

**Q15 fixed-point transforms** mirror the float pipeline with 16-bit fixed-point math, so the same operations run
on MCUs without a floating-point unit. Q15 maps ``[-1, +1]`` to ``[-32768, +32767]``.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Transform
     - Description
   * - ``TO_Q15``
     - Convert float to Q15 with saturation.
   * - ``FFT_Q15``
     - Fixed-point real FFT (equivalent of ``FFT_FE``).
   * - ``Q15_SCALE``
     - Bit-shift scaling by ``q15_scale_factor`` (positive amplifies, negative attenuates).
   * - ``Q15_MAG``
     - Magnitude of the complex Q15 FFT output (equivalent of ``ABS``).
   * - ``BIN_Q15``
     - Average per bin, optionally normalized by ``normalize_bin``, clipped to 16 bits (equivalent of ``BINNING``).

**Example: FFT with binning**

.. code-block:: yaml

   data_processing_feature_extraction:
     feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT']
     frame_size: 1024
     feature_size_per_frame: 64
     num_frame_concat: 4
     variables: 1

**Example: Q15 pipeline for MCUs**

.. code-block:: yaml

   data_processing_feature_extraction:
     feat_ext_transform: ['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT']
     frame_size: 256
     feature_size_per_frame: 16
     num_frame_concat: 8
     q15_scale_factor: 5
     normalize_bin: True
     variables: 1

Custom feature extraction
=========================

Use a ``Custom_*`` name and define the pipeline yourself:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     feature_extraction_name: 'Custom_Default'
     feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT']
     frame_size: 32
     feature_size_per_frame: 8
     num_frame_concat: 8
     variables: 5

Add the fine-grained keys (``normalize_bin``, ``stacking``, ``offset``, ``scale``, ``frame_skip``, ``log_mul``,
``log_base``, ``log_threshold``) when you need them.

Windowing and framing
=====================

Frame offset
------------

``offset`` controls the overlap between consecutive frames. ``offset: 0`` means no overlap. With ``offset: n`` the
step is ``1/n`` of a frame: ``offset: 2`` gives 50% overlap and ``offset: 4`` gives 75%. More overlap produces more
training samples from the same data.

.. code-block:: yaml

   data_processing_feature_extraction:
     offset: 2
     frame_size: 256
     variables: 3

Stacking modes
--------------

``stacking`` controls how multi-channel features are arranged for the model.

* ``'1D'``: features of all channels are concatenated into one flat sequence (``[a1, a2, a3, b1, b2, b3, ...]``). Good for fully connected models.
* ``'2D1'`` (default): a 2D matrix with one row per channel. Good for convolutional models.

Analysis bandwidth
------------------

``analysis_bandwidth: 1`` analyses the full spectrum; ``0.5`` only the lower half. Use it when only part of the
spectrum matters.

Gain variation augmentation
---------------------------

``gain_variations`` multiplies the raw data of each class by a random gain drawn from ``U(min, max)``, which makes the
model robust to amplitude changes. It is applied before feature extraction, with ``Downsample`` or ``SimpleWindow``.

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: [Downsample, SimpleWindow]
     gain_variations:
       arc: [0.9, 1.1]
       normal: [0.8, 1.2]
     sampling_rate: 313000
     new_sr: 3130
     frame_size: 256
     stride_size: 0.01
     variables: 1

Forecasting
-----------

Forecasting needs ``SimpleWindow`` plus the forecasting keys:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: [SimpleWindow]
     frame_size: 32
     stride_size: 0.1
     forecast_horizon: 2
     variables: 1
     target_variables: [0]

Audio front-ends
================

Audio classification (``target_module: audio``) selects the front end with ``audio_feature`` and a preset. The
audio parameters and their defaults are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``audio_feature``
     - ``MFCC`` (default), ``LPC``, ``FFT``, ``FB`` (learned filterbank) or ``RAW``.
   * - ``sampling_rate``, ``audio_duration_ms``
     - Audio sampling rate (default 16000) and clip length (default 1000 ms).
   * - ``frame_length_ms``, ``frame_step_ms``
     - Analysis frame length and step (defaults 30 and 20 ms).
   * - ``n_mfcc``, ``n_mels``
     - MFCC settings (defaults 10 and 40).
   * - ``nlpc``, ``lpc_order``
     - LPC settings (defaults 70 and 10).
   * - ``fb_conv_kernel``, ``fb_output_channel``, ``fb_conv_stride``, ``fb_context_ms``, ``fb_bitwidth``, ``input_bit_depth``
     - Learned filterbank settings (defaults 64, 64, 4, 20, 8 and 16).
   * - ``frame_size``, ``feature_size_per_frame``, ``num_frame_concat``, ``q15_scale_factor``, ``min_bin``
     - FFT front-end settings (defaults 512, 32, 1, 8 and 1).
   * - ``normalize_audio``, ``mono``
     - Normalize the waveform (default true) and mix to mono (default true).

Audio presets:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Preset
     - Front end
   * - ``GoogleSpeechCommands_MFCC_Default``
     - MFCC (10 coefficients, 40 mel bands).
   * - ``MSPM0_LPC_8k``, ``MSPM0_LPC_8820``
     - LPC front end that matches the MSPM0G5187 firmware pipeline (8000 Hz, and 8820 Hz from the I2S path), used by cough detection.
   * - ``WakeWordDetection_Filterbank_Default``
     - Learned filterbank: a Conv1D filter bank that is trained jointly with the model (``fb_*`` keys), used with ``TCDS_ResNet_FB_NPU``.
   * - ``GlassBreakDetection_512FFT_Default``
     - Q15 FFT, binning and concatenation (512 point FFT, 32 features per frame), used with ``DSCNN_GB_NPU``.

See :doc:`/task_types/audio_classification`, :doc:`/examples/wake_word_detection` and
:doc:`/examples/glass_break_detection`.

Radar and image inputs
======================

**Radar** classification does use the ``data_proc_transforms`` mechanism, but with its own radar stages instead of
the time series transforms above. The stages run in this order, and a preset or your YAML selects the ones it needs:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - ``data_proc_transforms`` entry
     - What it does
   * - ``Radar Range Data``
     - Selects FFT range bins (``bin*`` columns), optionally limited with ``radar_range_bins_bounds``. Used for
       surface classification.
   * - ``Radar Point Cloud Data``
     - Selects and filters point-cloud fields (``radar_point_cloud_data_selector``, coordinate system, spatial and
       non-spatial filter bounds).
   * - ``Radar Track Data``
     - Selects and filters tracker fields (``radar_track_data_selector``, velocity, acceleration and spatial bounds).
   * - ``Simple Window``
     - Stacks ``frame_size`` radar frames into one sample (``stride_size`` sets the overlap) and applies the frame
       quality checks (``radar_frames_quality_index``, ``radar_min_points_per_window``) and feature ordering
       (``radar_feature_ordering``).
   * - ``Scaling``
     - Scales values such as track velocity (``radar_data_scaling_track_velocity``).

The radar presets are ``Generic_248Input_Posture_and_Fall_Detection``, ``Generic_130Input_Radar_SurfaceClassification``,
``Generic_392Input_PointCloudWithTrack_SimpleWindow`` and ``Generic_Radar_SimpleWindowOnly``. See
:doc:`/task_types/radar_classification` and the full parameter reference in :doc:`/RADAR_PROCESSING_PIPELINE`.

**Image** classification is the one that uses image-style parameters: its presets (for example ``Mnist_Default``) set
``image_height``, ``image_width``, ``image_num_channel``, ``image_mean`` and ``image_scale``. See
:doc:`/task_types/image_classification`.

Evaluating feature extraction quality
=====================================

Before training, ModelMaker writes PCA plots of the extracted features for classification tasks
(``pca_on_feature_extracted_train_data.png`` and ``..._validation_data.png`` in the ``training/base`` folder of the run).
Clusters that are visibly separable are easier for the model to learn.

.. figure:: /_static/img/feature_extraction/pca_on_feature_extracted_train_data.png
   :width: 600px
   :align: center
   :alt: PCA on training data

   PCA of the extracted features on training data

.. figure:: /_static/img/feature_extraction/pca_on_feature_extracted_validation_data.png
   :width: 600px
   :align: center
   :alt: PCA on validation data

   PCA of the extracted features on validation data

* Tight, well-separated clusters: the features represent the classes well.
* Overlapping clusters: try a different feature extraction.
* Scattered points: high variance, possibly noisy data.

No single feature extraction is best for every dataset. Frequency-domain presets (FFT with binning) usually do
better on periodic signals like vibration, and raw presets on waveform-shape problems. Compare several presets.

Feature extraction only mode
----------------------------

Set ``dont_train_just_feat_ext: True`` to run only data processing and feature extraction. Combine with
``store_feat_ext_data: True`` to export the features. This is a fast way to compare presets by their PCA plots:

.. code-block:: yaml

   data_processing_feature_extraction:
     dont_train_just_feat_ext: True
     store_feat_ext_data: True
     feature_extraction_name: 'Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'
     variables: 3

Memory cost on the device
=========================

Feature extraction runs on the MCU before inference and needs its own RAM and Flash. Since 1.5.0 you can see
this cost before training: the compile step reports a *Feature Extraction Library* memory block next to the
model, and the standalone ``estimate_memory`` tool prints it in seconds. See :doc:`/features/memory_estimation`.

Best practices
==============

1. Match the transform to the signal: FFT for periodic signals, raw for transient ones.
2. Start from a standard preset and customize only if needed.
3. Mind the device: fewer features give smaller models and faster inference.
4. Compare several options with ``dont_train_just_feat_ext`` and the PCA plots.

Next steps
==========

* :doc:`/features/goodness_of_fit` to analyze dataset quality
* :doc:`/features/quantization` for model compression
* :doc:`/task_types/timeseries_classification`
