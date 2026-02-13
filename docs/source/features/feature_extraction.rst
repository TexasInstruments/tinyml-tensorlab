==================
Feature Extraction
==================

Feature extraction transforms raw sensor data into a representation that
helps the neural network learn patterns more effectively.

Overview
--------

Why use feature extraction?

* **Reduced input size**: Compress long time series
* **Better patterns**: Transform to domain where patterns are clearer
* **Faster inference**: Smaller inputs mean faster models
* **Domain knowledge**: Incorporate signal processing expertise

Feature Extraction Pipeline
---------------------------

Raw data flows through two stages: data processing transforms, then
feature extraction transforms:

.. code-block:: text

   Raw Signal → Data Processing → Feature Extraction → Model Input
                (data_proc_transforms)   (feat_ext_transform)
                e.g. SimpleWindow,       e.g. FFT_FE, BINNING,
                     Downsample               ABS, LOG_DB, CONCAT

Configuration Parameters
------------------------

The ``data_processing_feature_extraction`` section supports the following
parameters. There are two usage modes: using a **preset** name, or defining
a **custom** pipeline.

**Core Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``feature_extraction_name``
     - Preset name (e.g., ``'Generic_1024Input_FFTBIN_64Feature_8Frame'``)
       or a custom name starting with ``'Custom_'`` (e.g., ``'Custom_Default'``,
       ``'Custom_ArcFault'``). When using a preset, the transform pipeline is
       predefined. When using a ``Custom_*`` name, you must specify
       ``feat_ext_transform`` and related parameters.
   * - ``data_proc_transforms``
     - List of data processing transforms applied before feature extraction.
       Common values: ``['SimpleWindow']``, ``['Downsample']``,
       ``['SimpleWindow', 'Downsample']``, ``['Downsample', 'SimpleWindow']``,
       or ``[]`` (empty).
   * - ``feat_ext_transform``
     - List of feature extraction transforms applied in order. Common values
       include: ``'FFT_FE'``, ``'FFT_POS_HALF'``, ``'WINDOWING'``,
       ``'BINNING'``, ``'NORMALIZE'``, ``'ABS'``, ``'LOG_DB'``, ``'DC_REMOVE'``,
       ``'CONCAT'``, ``'FFT_Q15'``, ``'Q15_SCALE'``, ``'Q15_MAG'``,
       ``'BIN_Q15'``, ``'ECG_NORMALIZE'``.
   * - ``variables``
     - Number of input channels/variables. Supports three formats:
       an integer (select first N columns), a list of column indices
       (e.g., ``[0, 2, 4]``), or a list of column names
       (e.g., ``['accel_x', 'accel_y', 'accel_z']``).
   * - ``frame_size``
     - Number of samples per frame (e.g., ``128``, ``256``, ``512``, ``1024``).
   * - ``feature_size_per_frame``
     - Number of output features per frame after transform
       (e.g., ``8``, ``16``, ``32``, ``64``, ``128``).
   * - ``num_frame_concat``
     - Number of frames to concatenate (e.g., ``1``, ``4``, ``8``).
       Total features = ``feature_size_per_frame`` x ``num_frame_concat``.
   * - ``stride_size``
     - Stride between frames as a fraction (e.g., ``0.01``, ``0.1``, ``0.25``,
       ``0.5``, ``1``).

**Signal Processing Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``sampling_rate``
     - Original sampling rate of the input signal. Used with the
       ``Downsample`` data processing transform.
   * - ``new_sr``
     - Target sampling rate after downsampling. Used with the
       ``Downsample`` data processing transform.
   * - ``scale``
     - Scaling factor applied to input data (e.g., ``0.00390625`` for 1/256).
   * - ``offset``
     - Offset added to input data.
   * - ``frame_skip``
     - Number of frames to skip between selected frames (e.g., ``1``, ``8``).
   * - ``normalize_bin``
     - Enable bin normalization (``True``/``1``).
   * - ``stacking``
     - Feature stacking mode: ``'2D1'`` or ``'1D'``.
   * - ``min_bin``
     - Minimum frequency bin index to include.
   * - ``analysis_bandwidth``
     - Fraction of bandwidth to analyse (e.g., ``1`` for full bandwidth).

**Logarithmic Transform Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``log_mul``
     - Multiplier for logarithmic scaling (e.g., ``20`` for dB).
   * - ``log_base``
     - Base for logarithm (e.g., ``10``).
   * - ``log_threshold``
     - Minimum threshold to avoid log(0) (e.g., ``1e-100``).

**Fixed-Point (Q15) Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``q15_scale_factor``
     - Scale factor for Q15 fixed-point quantization (e.g., ``4``, ``5``).

**Data Augmentation and Testing:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``gain_variations``
     - Dictionary mapping class names to ``[min_gain, max_gain]`` ranges for
       data augmentation. Example:
       ``{arc: [0.9, 1.1], normal: [0.8, 1.2]}``.
   * - ``gof_test``
     - Run Goodness of Fit test on extracted features (``True``/``False``).

**Output Control:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``store_feat_ext_data``
     - Store extracted feature data to disk (``True``/``False``).
   * - ``nn_for_feature_extraction``
     - Use neural network for feature extraction (``True``/``False``).

**Forecasting-Specific Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``forecast_horizon``
     - Number of future timesteps to predict (e.g., ``1``, ``2``).
   * - ``target_variables``
     - List of column indices or names for the target variable(s) to forecast
       (e.g., ``[0]``, ``[5]``, ``['temperature']``).

Preset System
-------------

Tiny ML Tensorlab provides predefined feature extraction presets. When using
a preset, simply specify the ``feature_extraction_name`` and ``variables``:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'
     variables: 1

**Preset Naming Convention:**

.. code-block:: text

   Generic_<InputSize>Input_<Transform>_<Features>Feature_<Frames>Frame

   Example: Generic_1024Input_FFTBIN_64Feature_8Frame
   - Input: 1024 samples
   - Transform: FFT with binning
   - Features: 64 frequency bins
   - Frames: 8 temporal frames
   - Total: 64 x 8 = 512 features to model

Available Presets
-----------------

**FFT-Based Presets:**

Best for frequency-domain patterns (vibration, arc faults):

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Preset
     - Features
     - Use Case
   * - ``Generic_1024Input_FFTBIN_64Feature_8Frame``
     - 512
     - General purpose
   * - ``Generic_512Input_FFTBIN_32Feature_8Frame``
     - 256
     - Smaller input
   * - ``FFT1024Input_256Feature_1Frame_Full_Bandwidth``
     - 256
     - Full spectrum

**Raw Time-Domain Presets:**

Best for waveform shape patterns:

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - Preset
     - Features
     - Use Case
   * - ``Generic_512Input_RAW_512Feature_1Frame``
     - 512
     - Full waveform
   * - ``Generic_256Input_RAW_256Feature_1Frame``
     - 256
     - Shorter window
   * - ``Generic_128Input_RAW_128Feature_1Frame``
     - 128
     - Compact input

**Application-Specific Presets:**

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Preset
     - Application
   * - ``Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1``
     - Motor fault (3-axis)
   * - ``FFT1024Input_256Feature_1Frame_Full_Bandwidth``
     - Arc fault detection
   * - ``PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D``
     - PIR detection

Data Processing Transforms
---------------------------

The ``data_proc_transforms`` parameter specifies preprocessing steps applied
to raw data before feature extraction:

**SimpleWindow**

Segments continuous data into fixed-size windows:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 256
     stride_size: 0.01
     variables: 1

**Downsample**

Reduces the sampling rate of input data:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: ['Downsample', 'SimpleWindow']
     sampling_rate: 313000
     new_sr: 3130
     frame_size: 256
     stride_size: 0.01
     variables: 1

Multiple transforms can be chained in order:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     - Downsample
     frame_size: 256
     sampling_rate: 100
     new_sr: 1
     variables: 1

Feature Extraction Transforms
------------------------------

The ``feat_ext_transform`` parameter defines the feature extraction pipeline
as an ordered list of transforms. Each step processes the output of the
previous step.

**Common Transform Steps:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Transform
     - Description
   * - ``FFT_FE``
     - Compute FFT (Fast Fourier Transform)
   * - ``FFT_POS_HALF``
     - Keep only positive frequency half of FFT
   * - ``WINDOWING``
     - Apply windowing function
   * - ``BINNING``
     - Group frequency bins to reduce feature count
   * - ``NORMALIZE``
     - Normalize features
   * - ``ABS``
     - Take absolute value
   * - ``LOG_DB``
     - Convert to logarithmic (dB) scale
   * - ``DC_REMOVE``
     - Remove DC component
   * - ``CONCAT``
     - Concatenate frames into final feature vector
   * - ``FFT_Q15``
     - Fixed-point Q15 FFT (for MCU deployment)
   * - ``Q15_SCALE``
     - Q15 scaling
   * - ``Q15_MAG``
     - Q15 magnitude computation
   * - ``BIN_Q15``
     - Q15 binning
   * - ``ECG_NORMALIZE``
     - ECG-specific normalization

**Example: FFT with Binning Pipeline:**

.. code-block:: yaml

   data_processing_feature_extraction:
     feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT']
     frame_size: 1024
     feature_size_per_frame: 64
     num_frame_concat: 4
     variables: 1

**Example: FFT without Binning Pipeline:**

.. code-block:: yaml

   data_processing_feature_extraction:
     feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'LOG_DB', 'CONCAT']
     frame_size: 256
     feature_size_per_frame: 128
     num_frame_concat: 1
     variables: 6

**Example: Fixed-Point Q15 Pipeline (for MCU deployment):**

.. code-block:: yaml

   data_processing_feature_extraction:
     feat_ext_transform: ['FFT_Q15', 'Q15_SCALE', 'Q15_MAG', 'DC_REMOVE', 'BIN_Q15', 'CONCAT']
     frame_size: 256
     feature_size_per_frame: 16
     num_frame_concat: 8
     q15_scale_factor: 5
     normalize_bin: True
     variables: 1

Custom Feature Extraction
-------------------------

For advanced use cases, use a ``Custom_*`` feature extraction name and specify
the transform pipeline manually:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     feature_extraction_name: 'Custom_Default'
     feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'WINDOWING', 'BINNING', 'NORMALIZE', 'ABS', 'LOG_DB', 'CONCAT']
     frame_size: 32
     feature_size_per_frame: 8
     num_frame_concat: 8
     variables: 5

You can also configure additional parameters for fine-grained control:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: []
     feature_extraction_name: 'Custom_MotorFault'
     feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'BINNING', 'LOG_DB', 'CONCAT']
     frame_size: 1024
     feature_size_per_frame: 64
     num_frame_concat: 4
     normalize_bin: 1
     stacking: '1D'
     offset: 0
     scale: 1
     frame_skip: 1
     log_mul: 20
     log_base: 10
     log_threshold: 1e-100
     variables: 3

Multi-Channel Data
------------------

For sensors with multiple axes (e.g., 3-axis accelerometer), set
``variables`` to the number of channels:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'
     variables: 3

The ``variables`` parameter supports three formats:

* **Integer**: Select first N columns (e.g., ``variables: 3``)
* **List of indices**: Select specific columns (e.g., ``variables: [0, 2, 4]``)
* **List of names**: Select columns by name (e.g., ``variables: ['accel_x', 'accel_y', 'accel_z']``)

Forecasting Configuration
--------------------------

Forecasting tasks require specific additional parameters:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     frame_size: 32
     stride_size: 0.1
     forecast_horizon: 2
     variables: 1
     target_variables:
     - 0

.. note::

   ``SimpleWindow`` must be specified in ``data_proc_transforms`` for
   forecasting tasks.

Data Augmentation
-----------------

Use ``gain_variations`` to augment training data with gain variations per class:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
     - Downsample
     - SimpleWindow
     gain_variations:
       arc: [0.9, 1.1]
       normal: [0.8, 1.2]
     sampling_rate: 313000
     new_sr: 3130
     frame_size: 256
     stride_size: 0.01
     variables: 1

Choosing the Right Preset
-------------------------

**Decision Tree:**

.. code-block:: text

   Is the pattern in frequency content?
   |-- Yes --> Use FFT-based preset
   |   |-- Need full spectrum? --> FFT_FullBandwidth
   |   |-- Reduce features? --> FFTBIN
   |-- No --> Use RAW preset
       |-- Need temporal context? --> Multi-frame
       |-- Single snapshot? --> 1Frame

**Common Choices by Application:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Application
     - Recommended Preset
   * - Arc fault detection
     - ``FFT1024Input_256Feature_1Frame_Full_Bandwidth``
   * - Motor bearing fault
     - ``Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1``
   * - ECG classification
     - ``ECG2500Input_Roundoff_1Frame``
   * - Vibration anomaly
     - ``Generic_1024Input_FFTBIN_64Feature_8Frame``
   * - Simple waveforms
     - ``Generic_512Input_FFTBIN_32Feature_8Frame``
   * - PIR detection
     - ``PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D``

Performance Impact
------------------

Feature extraction affects model size and speed:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Features
     - Model Input
     - Model Size
     - Inference Time
   * - 128
     - Small
     - Smaller
     - Faster
   * - 256
     - Medium
     - Medium
     - Medium
   * - 512
     - Large
     - Larger
     - Slower

**Trade-off:**

* More features = more information = potentially better accuracy
* Fewer features = faster inference = fits smaller devices

On-Device Feature Extraction
----------------------------

Feature extraction runs on the MCU before inference. The compilation
process generates C code for the feature extraction pipeline configured
in your YAML.

**Memory Usage:**

Feature extraction buffers add to memory requirements:

.. code-block:: text

   Input buffer:  frame_size x variables x sizeof(data_type)
   FFT buffer:    frame_size x sizeof(data_type)
   Output buffer: feature_size_per_frame x num_frame_concat x sizeof(data_type)

Example Configurations
----------------------

**Arc Fault Classification (using preset):**

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

**Motor Bearing Fault (using preset with override):**

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'
     variables: 3
     feature_size_per_frame: 4

**Anomaly Detection with Downsampling:**

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     - Downsample
     frame_size: 1024
     sampling_rate: 100
     new_sr: 1
     variables: 1

**Regression with Simple Windowing:**

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     frame_size: 512
     stride_size: 0.1
     variables: 6

**Forecasting (PMSM Rotor Temperature):**

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     frame_size: 3
     stride_size: 0.4
     forecast_horizon: 1
     variables: 6
     target_variables:
     - 5

**Goodness of Fit Testing:**

Enable the ``gof_test`` parameter to run Goodness of Fit analysis on
extracted features:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'
     gof_test: True
     variables: 3

**PCA Visualization of Extracted Features:**

PCA (Principal Component Analysis) helps visualize how well the extracted features
separate your classes. Well-separated clusters indicate good feature extraction.

.. figure:: /_static/img/feature_extraction/pca_on_feature_extracted_train_data.png
   :width: 600px
   :align: center
   :alt: PCA on Training Data

   PCA visualization of extracted features on training data

.. figure:: /_static/img/feature_extraction/pca_on_feature_extracted_validation_data.png
   :width: 600px
   :align: center
   :alt: PCA on Validation Data

   PCA visualization of extracted features on validation data

**Interpreting PCA plots:**

* **Tight clusters**: Features represent the class well
* **Well-separated clusters**: Good class separability
* **Overlapping clusters**: May need different feature extraction
* **Scattered points**: High variance, potentially noisy data

Best Practices
--------------

1. **Match to signal characteristics**: FFT for periodic, raw for transient
2. **Start with standard presets**: Customize only if needed
3. **Consider device constraints**: Fewer features for smaller devices
4. **Test multiple options**: Compare accuracy with different presets
5. **Use domain knowledge**: Understand what patterns you're looking for

Next Steps
----------

* See :doc:`goodness_of_fit` to analyze dataset quality
* Learn about :doc:`quantization` for model compression
* Explore :doc:`/task_types/timeseries_classification` for classification
