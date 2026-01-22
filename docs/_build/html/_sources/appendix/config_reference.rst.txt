=========================
Configuration Reference
=========================

Complete reference for all YAML configuration options in Tiny ML Tensorlab.

Configuration File Structure
----------------------------

.. code-block:: yaml

   common:
     # General settings
   dataset:
     # Dataset configuration
   data_processing_feature_extraction:
     # Feature extraction settings
   training:
     # Training parameters
   testing:
     # Testing configuration
   compilation:
     # Compilation settings

Common Section
--------------

General project settings.

.. code-block:: yaml

   common:
     target_module: 'timeseries'    # 'timeseries' or 'image'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'
     run_name: '{date-time}/{model_name}'

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Required
     - Description
   * - ``target_module``
     - Yes
     - Module type: ``timeseries`` or ``image``
   * - ``task_type``
     - Yes
     - Task type (see below)
   * - ``target_device``
     - Yes
     - Target MCU device name
   * - ``run_name``
     - No
     - Output directory name pattern

**Task Types:**

* ``generic_timeseries_classification``
* ``generic_timeseries_regression``
* ``generic_timeseries_forecasting``
* ``generic_timeseries_anomalydetection``
* ``generic_image_classification``
* ``byom_compilation``

**Target Devices:**

* C2000: F28P55, F28P65, F29H85, F29P58, F29P32, F2837, F28004, F28003, F280013, F280015
* MSPM0: MSPM0G3507, MSPM0G3519, MSPM0G5187
* MSPM33C: MSPM33C32, MSPM33C34, AM13E2
* AM26x: AM263, AM263P, AM261
* Connectivity: CC2755, CC1352

Dataset Section
---------------

.. code-block:: yaml

   dataset:
     enable: True
     dataset_name: 'my_dataset'
     input_data_path: '/path/to/dataset'
     data_split_type: 'random'
     data_split_ratio: [0.8, 0.1, 0.1]

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``enable``
     - True
     - Enable dataset processing
   * - ``dataset_name``
     - Required
     - Dataset identifier
   * - ``input_data_path``
     - None
     - Path to custom dataset
   * - ``data_split_type``
     - 'random'
     - Split method: 'random', 'sequential', 'predefined'
   * - ``data_split_ratio``
     - [0.8, 0.1, 0.1]
     - Train/val/test split ratios
   * - ``input_data_split_type``
     - 'amongst_files'
     - 'amongst_files' or 'within_files'

Feature Extraction Section
--------------------------

.. code-block:: yaml

   data_processing_feature_extraction:
     enable: True
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'
     variables: 1
     frame_size: 1024
     gof_test: False

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``enable``
     - True
     - Enable feature extraction
   * - ``feature_extraction_name``
     - Required
     - Preset name or 'custom'
   * - ``variables``
     - 1
     - Number of input channels/variables
   * - ``frame_size``
     - 1024
     - Samples per frame
   * - ``num_frames``
     - 8
     - Number of temporal frames
   * - ``frame_stride``
     - frame_size/2
     - Stride between frames
   * - ``transform_type``
     - 'fft'
     - 'fft', 'fft_bin', 'raw', 'haar', 'hadamard'
   * - ``fft_size``
     - 1024
     - FFT size (if using FFT)
   * - ``num_bins``
     - 64
     - Number of frequency bins
   * - ``remove_dc``
     - False
     - Remove DC component
   * - ``normalization``
     - 'standard'
     - 'standard', 'minmax', 'none'
   * - ``gof_test``
     - False
     - Run Goodness of Fit test

**Forecasting-Specific:**

.. code-block:: yaml

   data_processing_feature_extraction:
     target_column: 0      # Column index to forecast
     forecast_horizon: 10  # Steps ahead to predict

Training Section
----------------

.. code-block:: yaml

   training:
     enable: True
     model_name: 'CLS_4k_NPU'
     training_epochs: 30
     batch_size: 256
     learning_rate: 0.001
     num_gpus: 0

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``enable``
     - True
     - Enable training
   * - ``model_name``
     - Required
     - Model name from registry
   * - ``training_epochs``
     - 20
     - Number of training epochs
   * - ``batch_size``
     - 256
     - Batch size for training
   * - ``learning_rate``
     - 0.001
     - Initial learning rate
   * - ``optimizer``
     - 'adam'
     - 'adam', 'sgd', 'adamw'
   * - ``weight_decay``
     - 0.0001
     - Weight decay (L2 regularization)
   * - ``num_gpus``
     - 0
     - Number of GPUs (0 for CPU)
   * - ``num_workers``
     - 4
     - Data loader workers
   * - ``seed``
     - 42
     - Random seed

**Quantization Options:**

.. code-block:: yaml

   training:
     quantization_type: 'int8'
     qat_enabled: True
     qat_start_epoch: 10
     ptq_calibration_samples: 500
     ptq_calibration_method: 'minmax'

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``quantization_type``
     - 'int8'
     - 'int8', 'int4', 'int2', 'mixed'
   * - ``qat_enabled``
     - False
     - Enable QAT
   * - ``qat_start_epoch``
     - 10
     - Epoch to start QAT
   * - ``ptq_calibration_samples``
     - 500
     - Samples for PTQ calibration
   * - ``ptq_calibration_method``
     - 'minmax'
     - 'minmax', 'histogram', 'entropy'

**Learning Rate Scheduler:**

.. code-block:: yaml

   training:
     lr_scheduler: 'cosine'
     lr_warmup_epochs: 5

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``lr_scheduler``
     - None
     - 'cosine', 'step', 'exponential'
   * - ``lr_warmup_epochs``
     - 0
     - Warmup epochs
   * - ``lr_step_size``
     - 10
     - Epochs per step (step scheduler)
   * - ``lr_gamma``
     - 0.1
     - LR decay factor

Testing Section
---------------

.. code-block:: yaml

   testing:
     enable: True
     test_float: True
     test_quantized: True

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``enable``
     - True
     - Enable testing
   * - ``test_float``
     - True
     - Test float32 model
   * - ``test_quantized``
     - True
     - Test quantized model
   * - ``save_predictions``
     - False
     - Save prediction results
   * - ``error_analysis``
     - False
     - Save misclassified samples

NAS Section
-----------

.. code-block:: yaml

   nas:
     enable: True
     search_type: 'multi_trial'
     num_trials: 20
     param_range: [500, 5000]
     accuracy_target: 0.95

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``enable``
     - False
     - Enable NAS
   * - ``search_type``
     - 'multi_trial'
     - 'single_trial', 'multi_trial'
   * - ``num_trials``
     - 20
     - Architectures to evaluate
   * - ``param_range``
     - [500, 5000]
     - [min, max] parameters
   * - ``accuracy_target``
     - 0.9
     - Minimum accuracy target
   * - ``npu_compatible``
     - True
     - Enforce NPU constraints

Compilation Section
-------------------

.. code-block:: yaml

   compilation:
     enable: True
     preset_name: 'compress_npu_layer_data'

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``enable``
     - True
     - Enable compilation
   * - ``preset_name``
     - 'default_preset'
     - Compilation preset
   * - ``optimize_memory``
     - True
     - Memory optimization
   * - ``debug_info``
     - False
     - Include debug symbols

**Compilation Presets:**

* ``default_preset`` - Standard compilation
* ``compress_npu_layer_data`` - NPU-optimized

BYOM Section
------------

For compilation-only mode:

.. code-block:: yaml

   byom:
     enable: True
     onnx_model_path: '/path/to/model.onnx'
     input_shape: [1, 1, 512, 1]
     already_quantized: False

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``enable``
     - False
     - Enable BYOM mode
   * - ``onnx_model_path``
     - Required
     - Path to ONNX model
   * - ``input_shape``
     - Required
     - Model input shape
   * - ``already_quantized``
     - False
     - True if model is quantized

Complete Example
----------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'
     run_name: '{date-time}/{model_name}'

   dataset:
     enable: True
     dataset_name: 'dc_arc_fault_example_dsk'
     input_data_path: 'https://software-dl.ti.com/...'
     data_split_type: 'random'
     data_split_ratio: [0.8, 0.1, 0.1]

   data_processing_feature_extraction:
     enable: True
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1
     gof_test: False

   training:
     enable: True
     model_name: 'ArcFault_model_400_t'
     training_epochs: 30
     batch_size: 256
     learning_rate: 0.001
     num_gpus: 0
     quantization_type: 'int8'
     qat_enabled: True
     qat_start_epoch: 15

   testing:
     enable: True
     test_float: True
     test_quantized: True

   compilation:
     enable: True
     preset_name: 'compress_npu_layer_data'
