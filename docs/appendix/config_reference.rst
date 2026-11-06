=========================
Configuration Reference
=========================

Reference for the YAML configuration options of Tiny ML ModelZoo 1.5.0. For a step-by-step guide on writing or
adapting a config, see :doc:`/CONFIG_YAML_GUIDE`.

.. note::

   The defaults below are read from the ``init_params()`` functions of ModelMaker (package ``tinyml_modelmaker``,
   file ``tinyml_modelmaker/ai_modules/<module>/params.py``). They are the values used when a key is missing from your
   YAML. The tables are written for the timeseries module (``target_module: timeseries``); the audio, radar and
   vision modules add or drop a few keys, listed at the end of each section. Keys that are filled in at run time
   (paths, model ids) are not meant to be set by the user and are listed as "internal".

How a YAML is merged
--------------------

The user YAML is loaded with ``yaml.safe_load`` and merged recursively on top of the defaults (``ConfigDict`` in
``tinyml_modelmaker/utils/config_dict.py``), so you only need to write the keys you want to change. Then, in the
ModelMaker runner:

* the ``training`` defaults of the chosen model are taken from its model description (``model_name`` must be in the
  ``enabled_models_list`` of the model descriptions; see :doc:`model_zoo_reference`),
* a ``feature_extraction_name`` preset, if given, fills the ``data_processing_feature_extraction`` keys it defines,
* a compilation preset is chosen from ``compile_preset_name`` and ``target_device``.

If CUDA is available, ``training.compile_model`` and ``training.native_amp`` are switched on automatically unless
you set them in the YAML or set ``training.num_gpus: 0`` (``utils/hardware_defaults.py``).

File structure
--------------

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

There is also a ``download`` key in the defaults (default ``None``). It is not documented in the examples and is not used
in the shipped configs.

Common section
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Key
     - Default
     - Description
   * - ``target_module``
     - ``'timeseries'``
     - ``timeseries``, ``audio``, ``radar`` or ``image`` (vision module).
   * - ``task_type``
     - ``None`` (required)
     - See the task type list below.
   * - ``task_category``
     - ``None``
     - Filled in from ``task_type``.
   * - ``target_device``
     - ``None`` (required)
     - Target MCU, for example ``F28P55``. See the device list below.
   * - ``run_name``
     - ``{date-time}/{model_name}``
     - Run folder pattern under ``projects_path``/<project>/run. ``{date-time}`` becomes
       ``%Y%m%d-%H%M%S`` and ``{model_name}`` the model name.
   * - ``download_path``
     - ``./data/downloads``
     - Where datasets are downloaded.
   * - ``projects_path``
     - ``./data/projects``
     - Parent folder of project outputs.
   * - ``project_path``, ``project_run_path``
     - ``None``
     - Internal, derived from ``projects_path`` and ``run_name``.
   * - ``generic_model``
     - ``False``
     - Set from the model description.

**Task types** (``common.task_type``):

* Timeseries: ``generic_timeseries_classification``, ``generic_timeseries_regression``,
  ``generic_timeseries_forecasting``, ``generic_timeseries_anomalydetection``
* Timeseries application aliases mapped to classification: ``arc_fault``, ``motor_fault``, ``ecg_classification``,
  ``blower_imbalance``, ``pir_detection``
* Other modules: ``audio_classification``, ``radar_point_cloud_classification``, ``radar_surface_classification``, ``image_classification``

The ``byom_compilation`` task type and the ``BYOM`` section of the 1.4.0 reference are not present in the 1.5.0 code
(no match in ``tinyml_modelmaker``). To use your own model see :doc:`/BRING_YOUR_OWN_MODEL`.

**Target devices** (``tinyml_modelmaker/ai_modules/timeseries/constants.py``, ``TARGET_DEVICES``; the exact devices
valid for a task depend on the model, see :doc:`model_zoo_reference`):

* C2000 F28x: F280013, F280015, F28003, F28004, F2837, F28P55, F28P65, F28E12, F2807x, F28002x, F28P551x, F2837xS, F2838x
* C2000 F29x: F29H85, F29P58, F29P32
* MSPM0 and MSPM33: MSPM0G3507, MSPM0G3519, MSPM0G5187, MSPM33C32
* AM13: AM13E2
* AM26x: AM263, AM263P, AM261
* Connectivity: CC2755, CC2745, CC1352, CC1312, CC1354, CC1314, CC35X1

Dataset section
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Key
     - Default
     - Description
   * - ``enable``
     - ``True``
     - Enable dataset handling.
   * - ``dataset_name``
     - ``None``
     - Name of a built-in dataset, or your own name when ``input_data_path`` is set.
   * - ``input_data_path``
     - ``None``
     - Path or URL of the dataset (folder, zip or URL). Used for custom data.
   * - ``input_annotation_path``
     - ``None``
     - Annotation file, if the format needs one.
   * - ``dataset_path``, ``extract_path``
     - ``None``
     - Internal; where the split dataset is created.
   * - ``split_factor``
     - ``(0.6, 0.3, 0.1)``
     - Train/val/test ratios.
   * - ``split_names``
     - ``('train', 'val', 'test')``
     - Names of the splits.
   * - ``split_type``
     - ``'amongst_files'``
     - ``amongst_files`` (whole files go to one split) or ``within_files`` (each file is split). Other values are not
       handled in ``ai_modules/common/datasets/__init__.py``.
   * - ``max_num_files``
     - ``10000``
     - Maximum number of files used.
   * - ``data_dir``
     - ``None``
     - ``None`` auto-detects the folder based on the task category.
   * - ``annotation_format``
     - ``'univ_ts_json'``
     - ``univ_ts_json`` or ``coco_json`` (any other value raises ``Unsupported annotation_format``).
   * - ``annotation_dir``
     - ``'annotations'``
     - Annotation folder name.
   * - ``annotation_prefix``
     - ``'instances'``
     - Annotation file prefix.
   * - ``input_data_path`` splits: ``data_path_splits``, ``annotation_path_splits``
     - ``None``
     - Internal.

Data processing and feature extraction section
----------------------------------------------

The preset in ``feature_extraction_name`` (see :doc:`/features/feature_extraction`) sets most of these keys
for you. Values you write in the YAML override the preset.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Key
     - Default
     - Description
   * - ``feature_extraction_name``
     - ``None``
     - Preset name, for example ``Generic_1024Input_FFTBIN_64Feature_8Frame``, or ``Custom_<name>`` for your own
       transform chain.
   * - ``data_proc_transforms``
     - ``[]``
     - Data processing transforms: ``SimpleWindow``, ``Downsample`` (combined as ``['Downsample', 'SimpleWindow']``).
   * - ``feat_ext_transform``
     - ``[]``
     - Feature extraction chain. Names used by the 1.5.0 presets: ``FFT_FE``, ``FFT_POS_HALF``, ``WINDOWING``,
       ``BINNING``, ``NORMALIZE``, ``ABS``, ``LOG_DB``, ``CONCAT``, ``DC_REMOVE``, ``RAW_FE``, ``FFT_Q15``,
       ``Q15_MAG``, ``Q15_SCALE``, ``BIN_Q15``, ``FE_Q15``, ``FE_DECIMATE``, ``FFT_COL``, ``L2_NORM_ROW``,
       ``PIR_FE``, ``PIR_FE_Q15``, ``ROUND_OFF``.
   * - ``variables``
     - ``1``
     - Input columns after the time column(s): an int (first n columns), a list of ints, or a list of column names.
   * - ``target_variables``
     - ``[]``
     - Forecasting: columns to predict.
   * - ``sampling_rate``
     - ``1``
     - Original sampling rate.
   * - ``new_sr``
     - ``1``
     - Target sampling rate (used with ``Downsample``).
   * - ``resampling_factor``
     - ``1``
     - Resampling factor.
   * - ``frame_size``
     - ``1``
     - Samples per frame.
   * - ``stride_size``
     - ``0.01``
     - Stride between frames as a fraction of ``frame_size``.
   * - ``forecast_horizon``
     - ``1``
     - Forecasting: number of future time steps to predict.
   * - ``feature_size_per_frame``
     - ``None``
     - Output features per frame.
   * - ``num_frame_concat``
     - ``1``
     - Frames concatenated into one input.
   * - ``frame_skip``
     - ``1``
     - Frames skipped between selected frames.
   * - ``min_bin``
     - ``1``
     - First FFT bin kept.
   * - ``analysis_bandwidth``
     - ``1``
     - Bandwidth factor for binning.
   * - ``normalize_bin``
     - ``0``
     - Bin normalization.
   * - ``dc_remove``
     - ``True``
     - Remove the DC component.
   * - ``q15_scale_factor``
     - ``4``
     - Q15 scaling for the fixed-point chain.
   * - ``log_base``, ``log_mul``, ``log_threshold``
     - ``None``
     - Parameters of ``LOG_DB``; the presets set ``log_mul=10``, ``log_base='e'``, ``log_threshold=1e-12``.
   * - ``stacking``
     - ``'2D1'``
     - Feature stacking mode (``2D1`` or ``1D``).
   * - ``offset``, ``scale``
     - ``0``, ``None``
     - Offset and scale applied to the input.
   * - ``gain_variations``
     - ``None``
     - Dict of class-to-gain range for augmentation.
   * - ``fft_size``, ``window_count``, ``chunk_size``
     - ``1``
     - Used by some presets.
   * - ``store_feat_ext_data``, ``feat_ext_store_dir``
     - ``False``, ``None``
     - Store extracted features to disk.
   * - ``dont_train_just_feat_ext``
     - ``False``
     - Only run feature extraction.
   * - ``nn_for_feature_extraction``
     - ``False``
     - Run feature extraction in the NN (for devices without the FE library).
   * - ``gof_test``
     - ``False``
     - Goodness-of-fit test.

Audio module (``target_module: audio``) replaces the timeseries keys with: ``sampling_rate`` (16000),
``audio_duration_ms`` (1000), ``audio_feature`` (``"MFCC"``), ``n_mfcc`` (10), ``n_mels`` (40), ``frame_length_ms``
(30), ``frame_step_ms`` (20), ``nlpc`` (70), ``lpc_order`` (10), the learned filterbank keys ``fb_conv_kernel`` (64),
``fb_output_channel`` (64), ``fb_bitwidth`` (8), ``fb_conv_stride`` (4), ``fb_context_ms`` (20), ``input_bit_depth``
(16), ``frame_size`` (512), ``normalize_audio`` (``True``), ``mono`` (``True``) and ``augmentation_transform``
(``[]``). The allowed values of ``audio_feature`` were not listed here; see ``ai_modules/audio/constants.py``.

Radar module (``target_module: radar``) has the timeseries keys ``sampling_rate`` (1), ``frame_size`` (8),
``variables`` (1), ``feat_ext_transform`` and ``data_proc_transforms``, and no feature extraction presets.

Vision module (``target_module: image``): ``image_height`` (28), ``image_width`` (28), ``image_num_channel`` (1),
``image_mean`` (0.1307), ``image_scale`` (0.3081), ``feat_ext_transform``, ``data_proc_transforms``,
``augmentation_transform`` and the transform parameters ``pad_value`` (0), ``binary_threshold`` (128),
``clahe_clip_limit`` (2.0), ``clahe_tile_grid_size`` ((8, 8)), ``sobel_mode`` (``"magnitude"``), ``sobel_ksize`` (3),
``laplacian_ksize`` (3), ``horizontal_flip_prob`` (0.5), ``vertical_flip_prob`` (0.5), ``random_rotation_deg`` (15) and
``color_jitter_brightness`` (0.10), ``_contrast`` (0.10), ``_saturation`` (0.05), ``_hue`` (0.01).

Training section
----------------

Training defaults are overridden per model by the model description (for example ``learning_rate`` and
``batch_size``), then by your YAML.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Key
     - Default
     - Description
   * - ``enable``
     - ``True``
     - Run the training stage.
   * - ``model_name``
     - ``None`` (required)
     - Must be listed in the ``enabled_models_list`` of the model descriptions. See :doc:`model_zoo_reference`.
   * - ``custom_model_path``
     - ``None``
     - Path to a Python file with your own model class, named exactly as ``model_name``. When set, ``model_name`` does
       not have to be in ``enabled_models_list``. The path must exist. Used by the BYOM quick start; see
       :doc:`/BRING_YOUR_OWN_MODEL`.
   * - ``model_spec``, ``model_config``
     - ``None``
     - Model specification or config (for spec-based models). ``model_spec`` is set automatically from
       ``custom_model_path``.
   * - ``dataset_loader``
     - ``'GenericTSDataset'``
     - ``GenericAudioDataset``, ``GenericImageDataset`` and ``GenericRadarDataset`` for the other modules.
   * - ``augment_config``
     - ``None``
     - Augmentation config. Usage not verified in this reference.
   * - ``pretrained_checkpoint_path``
     - ``None``
     - Start from this checkpoint.
   * - ``load_saved_model``
     - ``None``
     - Load a saved model. Usage not verified in this reference.
   * - ``training_epochs``
     - ``10``
     - Number of epochs.
   * - ``batch_size``
     - ``8``
     - Batch size (model description default overrides).
   * - ``learning_rate``
     - ``2e-3``
     - Learning rate.
   * - ``optimizer``
     - ``'sgd'``
     - ``sgd`` (``sgd_nesterov`` style names starting with ``sgd`` are accepted), ``rmsprop``, ``adam``, ``adamw``.
       Unknown values log a warning and fall back to Adam.
   * - ``weight_decay``
     - ``1e-4``
     - Weight decay.
   * - ``lr_scheduler``
     - ``'cosineannealinglr'``
     - ``cosineannealinglr``, ``steplr``, ``exponentiallr``, ``none``. Other values raise
       ``Invalid lr scheduler``.
   * - ``lambda_reg``
     - ``0``
     - Regularization weight.
   * - ``early_stopping``
     - ``True``
     - New in 1.5.0. Stop when the validation metric stops improving.
   * - ``early_stopping_patience``
     - ``5``
     - New in 1.5.0. Epochs without improvement before stopping.
   * - ``early_stopping_min_delta``
     - ``0.0``
     - New in 1.5.0. Minimum change that counts as improvement.
   * - ``training_device``
     - ``'cuda'``
     - ``cpu``, ``cuda`` or ``mps``. Falls back if unavailable.
   * - ``num_gpus``
     - ``1``
     - ``0`` or ``1``. ``0`` also disables the automatic ``compile_model``/``native_amp``.
   * - ``distributed``
     - ``True``
     - Use distributed training when launched for it.
   * - ``compile_model``
     - ``0``
     - ``1`` enables ``torch.compile``. Turned on automatically when CUDA is available and the key is not set.
   * - ``native_amp``
     - ``False``
     - PyTorch native AMP. Turned on automatically when CUDA is available and the key is not set.
   * - ``quantization``
     - ``0`` (``NO_QUANTIZATION``)
     - ``0`` none, ``1`` generic, ``2`` TINPU (NPU style). Many model descriptions default to ``2``.
   * - ``quantization_method``
     - ``'QAT'``
     - ``QAT`` or ``PTQ``.
   * - ``quantization_weight_bitwidth``
     - ``8``
     - Choices in the training script: 16, 8, 4, 2.
   * - ``quantization_activation_bitwidth``
     - ``8``
     - Activation bitwidth.
   * - ``auto_quantization``
     - ``True``
     - New in 1.5.0. Automatic bitwidth selection (binary search, Hessian-based mixed precision). Set ``False`` to
       use the fixed bitwidths above.
   * - ``autoquant_tolerance_classification``
     - ``0.05``
     - Tolerated accuracy drop (fraction) versus the float model.
   * - ``autoquant_tolerance_regression``
     - ``0.05``
     - Tolerated R2 drop (fraction).
   * - ``autoquant_tolerance_forecasting``
     - ``2.0``
     - Tolerated SMAPE increase: float SMAPE x (1 + 2.0).
   * - ``autoquant_tolerance_anomaly``
     - ``2.0``
     - Tolerated MSE increase: float MSE x (1 + 2.0).
   * - ``output_int``
     - ``None``
     - ``None`` decides automatically from the task category and quantization (integer output for classification
       when quantized, float output for regression, forecasting and anomaly detection). ``True``/``False`` forces
       it and overrides the compile target string.
   * - ``dual_op``
     - ``False``
     - Passed to the radar training script as ``--dual-op``; unused by the other timeseries tasks in this reference.
   * - ``run_quant_train_only``
     - ``False``
     - Run only the quantization training step.
   * - ``ondevice_training``
     - ``False``
     - Enable the on-device training export (frozen model plus trainable model configuration and data files). Float ODT uses ``quantization: 0``, integer ODT requires ``quantization: 2``. See :doc:`/features/ondevice_training`.
   * - ``trainable_layers_from_last``
     - ``1``
     - Number of main layers (Linear/Conv) from the end that are trainable on device. Activation and reshape layers do not count.
   * - ``target_device_flash_kb``
     - ``100``
     - Flash size of the target in KB.
   * - ``export_samples_per_class``
     - ``(10, 10, 10)``
     - Samples exported per class for (train, val, test), embedded as C arrays in the on-device training data files (``ondevice_training_data.c`` for float ODT, ``train_data.c`` for integer ODT).
   * - ``nas_enabled``
     - ``False``
     - Neural architecture search.
   * - ``nas_optimization_mode``
     - ``'Memory'``
     - ``Memory`` or ``Compute``.
   * - ``nas_epochs``
     - ``10``
     - Search epochs.
   * - ``nas_model_size``
     - ``None``
     - Preset mode: ``s``, ``m``, ``l``, ``xl``.
   * - ``nas_nodes_per_layer``, ``nas_layers``, ``nas_init_channels``, ``nas_init_channel_multiplier``, ``nas_fanout_concat``
     - ``4``, ``3``, ``1``, ``3``, ``4``
     - Customization mode. ``nas_layers`` minimum is 3; ``nas_fanout_concat`` must not exceed ``nas_nodes_per_layer``.
   * - ``model_training_id``, ``training_backend``, ``target_devices``, ``properties``
     - various
     - Filled from the model description (internal / GUI metadata).
   * - ``project_path``, ``dataset_path``, ``training_path``, ``training_path_quantization``, ``train_output_path``, ``log_file_path``, ``file_level_classification_log_path``, ``log_summary_regex``
     - ``None``
     - Internal.

The radar module also has ``momentum`` (default ``0``).

Memory estimation (pre-flight) is not a YAML key. The training script has a ``--skip_memory_preflight`` flag, and the
standalone ``estimate_memory`` command takes a config file; see :doc:`/features/memory_estimation`.

Testing section
---------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Key
     - Default
     - Description
   * - ``enable``
     - ``True``
     - Run the test stage.
   * - ``skip_train``
     - ``False``
     - Skip training and test an existing model (set ``model_path``).
   * - ``model_path``
     - ``None``
     - Model to test when training is skipped.
   * - ``test_data``
     - ``None``
     - Test data location.
   * - ``device_inference``
     - ``False``
     - Run inference on the device. Usage not verified in this reference.

Compilation section
-------------------

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Key
     - Default
     - Description
   * - ``enable``
     - ``True``
     - Run the compilation stage (needs the compiler tools, see :doc:`/troubleshooting/common_errors`).
   * - ``compile_preset_name``
     - ``None``, then ``default_preset``
     - ``default_preset``, ``compress_npu_layer_data`` (NPU, optimizes layer data for space) or
       ``forced_soft_npu_preset`` (use the software NPU on an NPU device such as F28P55).
   * - ``keep_libc_files``
     - ``False``
     - Keep intermediate library files.
   * - ``model_path``
     - ``None``
     - Model to compile (normally set by the pipeline).
   * - ``compile_output_path``
     - ``None``
     - Internal.

See :doc:`/devices/index` and :doc:`/NPU_CONFIGURATION_GUIDELINES` for which preset suits which device.

Example
-------

This is ``examples/dc_arc_fault/config_dsk.yaml`` without comments:

.. code-block:: yaml

   common:
     task_type: arc_fault
     target_device: F28P55
   dataset:
     dataset_name: dc_arc_fault_example_dsk
   data_processing_feature_extraction:
     feature_extraction_name: FFT1024Input_256Feature_1Frame_Full_Bandwidth
     variables: 1
   training:
     model_name: CLS_1k_NPU
     model_config: ''
     batch_size: 32
     training_epochs: 10
     num_gpus: 0
     early_stopping: true
     early_stopping_patience: 5
   testing: {}
   compilation: {}
