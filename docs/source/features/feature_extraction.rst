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

Raw data flows through these stages:

.. code-block:: text

   Raw Signal → Framing → Transform → Binning → Concatenation → Model Input
        │           │         │          │            │
     1024 samples   │      FFT/Raw     64 bins    8 frames = 512 features
                 128 samples
                 per frame

Preset System
-------------

Tiny ML Tensorlab provides predefined feature extraction presets:

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
   - Total: 64 × 8 = 512 features to model

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
   * - ``Generic_128Input_RAW_128Feature_4Frame``
     - 512
     - Temporal context

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

Transform Types
---------------

**FFT (Fast Fourier Transform)**

Converts time domain to frequency domain:

.. code-block:: yaml

   data_processing_feature_extraction:
     transform_type: 'fft'
     fft_size: 1024

Use when:

* Patterns are in frequency content
* Signal is periodic
* Looking for harmonic signatures

**FFT with Binning**

Groups FFT bins for reduced feature count:

.. code-block:: yaml

   data_processing_feature_extraction:
     transform_type: 'fft_bin'
     fft_size: 1024
     num_bins: 64

Benefits:

* Reduces feature count
* Smooths frequency representation
* More robust to frequency shifts

**Raw (No Transform)**

Uses time-domain samples directly:

.. code-block:: yaml

   data_processing_feature_extraction:
     transform_type: 'raw'

Use when:

* Waveform shape is important
* Transient patterns
* Phase information matters

**Wavelet Transforms**

Time-frequency representation:

.. code-block:: yaml

   data_processing_feature_extraction:
     transform_type: 'haar'  # or 'hadamard'

Use when:

* Need both time and frequency info
* Transient events
* Multi-scale analysis

Multi-Frame Configuration
-------------------------

Multiple frames provide temporal context:

.. code-block:: yaml

   data_processing_feature_extraction:
     frame_size: 128
     num_frames: 8
     frame_stride: 64  # 50% overlap

**How it works:**

.. code-block:: text

   Signal: [==========================================]
                                   1024 samples

   Frame 1: [====]
   Frame 2:    [====]
   Frame 3:       [====]
   ...
   Frame 8:                                      [====]

   Each frame: 128 samples → transform → 64 features
   Total: 64 × 8 = 512 features

Multi-Channel Data
------------------

For sensors with multiple axes (e.g., 3-axis accelerometer):

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'
     variables: 3

**Processing:**

Each channel is processed independently, then concatenated:

.. code-block:: text

   X-axis: 128 features
   Y-axis: 128 features
   Z-axis: 128 features
   Total:  384 features

DC Removal
----------

Remove DC offset for cleaner frequency analysis:

.. code-block:: yaml

   data_processing_feature_extraction:
     remove_dc: True

This is important for:

* AC-coupled signals
* Vibration data
* When absolute level doesn't matter

Normalization
-------------

Normalize features for consistent model input:

.. code-block:: yaml

   data_processing_feature_extraction:
     normalization: 'standard'  # or 'minmax', 'none'

**Options:**

* ``standard``: Zero mean, unit variance
* ``minmax``: Scale to [0, 1]
* ``none``: No normalization

Custom Feature Extraction
-------------------------

For advanced use cases, define custom extraction:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'custom'
     custom_config:
       frame_size: 256
       frame_stride: 128
       num_frames: 4
       transform: 'fft'
       fft_size: 256
       num_bins: 32
       remove_dc: True
       normalization: 'standard'

Choosing the Right Preset
-------------------------

**Decision Tree:**

.. code-block:: text

   Is the pattern in frequency content?
   ├── Yes → Use FFT-based preset
   │   └── Need full spectrum? → FFT_FullBandwidth
   │   └── Reduce features? → FFTBIN
   └── No → Use RAW preset
       └── Need temporal context? → Multi-frame
       └── Single snapshot? → 1Frame

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
     - ``Generic_512Input_RAW_512Feature_1Frame``
   * - Vibration anomaly
     - ``Generic_1024Input_FFTBIN_64Feature_8Frame``
   * - Simple waveforms
     - ``Generic_512Input_FFTBIN_32Feature_8Frame``

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

Feature extraction runs on the MCU before inference:

**Code Generated:**

The compilation process generates C code for feature extraction:

.. code-block:: c

   // Generated feature extraction
   void extract_features(float* input, float* features) {
       // FFT
       // Binning
       // Normalization
   }

**Memory Usage:**

Feature extraction buffers add to memory requirements:

.. code-block:: text

   Input buffer:  frame_size × variables × 4 bytes
   FFT buffer:    fft_size × 4 bytes
   Output buffer: num_features × 4 bytes

Example Configuration
---------------------

**Complete feature extraction setup:**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'motor_fault_classification_dsk'

   data_processing_feature_extraction:
     feature_extraction_name: 'Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'
     variables: 3
     remove_dc: True
     normalization: 'standard'

   training:
     model_name: 'MotorFault_model_1_t'
     training_epochs: 20

Debugging Feature Extraction
----------------------------

To visualize extracted features:

.. code-block:: yaml

   data_processing_feature_extraction:
     visualize: True
     save_examples: 10  # Save first 10 samples

This generates:

* Feature visualizations
* Before/after transform comparisons
* Statistics about feature distributions

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
