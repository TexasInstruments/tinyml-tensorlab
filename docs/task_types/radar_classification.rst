====================
Radar Classification
====================

Radar classification assigns a label to windows of radar data from a TI mmWave sensor. Two task types are
available: **point cloud classification** (for example human posture and fall detection) and **surface
classification** (for example grass vs not grass, from 1-D FFT range profiles). Radar tasks arrived in
Tiny ML ModelZoo 1.5.0; surface classification, the dual-stream model and the radar processing pipeline guide
are new in 1.6.0.

Overview
--------

**What it does**: Takes a window of consecutive radar frames and outputs a class label.

**Examples**:

* Point cloud and track data, 8 frames -> "standing", "falling", ... (:doc:`/examples/radar_posture_and_fall_detection`)
* FFT range bins, 5 frames x 26 bins -> "grass" or "not grass" (:doc:`/examples/radar_surface_classification`)

The full data pre-processing chain (column names, presets, every configurable parameter) is described in
:doc:`/RADAR_PROCESSING_PIPELINE`.

Task Settings
-------------

.. list-table::
   :widths: 30 70

   * - ``task_type``
     - ``radar_point_cloud_classification`` or ``radar_surface_classification``
       (both belong to the task category ``radar_classification``)
   * - ``target_module``
     - ``radar``
   * - Target device
     - ``xWRL6432`` (the only radar target device; compiled with the TI Arm Clang toolchain)
   * - Data folder
     - ``classes/<class_name>/*.csv`` (or ``*.npy``)
   * - Feature extraction presets
     - ``Generic_248Input_Posture_and_Fall_Detection`` (point cloud + track, 8 frames, 248 inputs),
       ``Generic_130Input_Radar_SurfaceClassification`` (range bins 6-31, 5 frames, 130 inputs),
       ``Generic_392Input_PointCloudWithTrack_SimpleWindow`` (full 49 features x 8 frames) and
       ``Generic_Radar_SimpleWindowOnly`` (windowing only, for pre-processed data)
   * - Early stopping
     - On by default (``training.early_stopping``, ``early_stopping_patience: 5``);
       see :doc:`/features/early_stopping`

Configuration
-------------

Point cloud classification, from ``examples/radar_posture_and_fall_detection/config.yaml``:

.. code-block:: yaml

   common:
     task_type: radar_point_cloud_classification
     target_module: radar
     target_device: xWRL6432

   dataset:
     dataset_name: radar_point_cloud_classification
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/posture_and_fall_detection_dataset.zip
     split_type: within_files
     split_factor: [0.7, 0.2, 0.1]

   data_processing_feature_extraction:
     feature_extraction_name: "Generic_248Input_Posture_and_Fall_Detection"
     data_proc_transforms: ['Radar Point Cloud Data', 'Radar Track Data', 'Simple Window', 'Scaling']
     radar_data_scaling_track_velocity: 30

   training:
     model_name: RF_DualStream_PointCloud_19K
     training_epochs: 250
     learning_rate: 0.001
     optimizer: adamw
     weight_decay: 0
     lr_scheduler: CosineAnnealingLR
     num_gpus: 0

   testing: {}
   compilation: {}

Surface classification, from ``examples/radar_surface_classification/config.yaml``:

.. code-block:: yaml

   common:
     task_type: radar_surface_classification
     target_module: radar
     target_device: xWRL6432

   dataset:
     dataset_name: radar_surface_classification
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/radar_grass_surface_dataset.zip
     split_type: amongst_files
     split_factor: [0.7, 0.2, 0.1]

   data_processing_feature_extraction:
     feature_extraction_name: Generic_130Input_Radar_SurfaceClassification
     frame_size: 5
     stride_size: 0.2
     radar_range_bins_bounds: [null, null]   # [null, null] keeps all range bins; [6, 31] selects bins 6-31

   training:
     model_name: RF_MLP_3L_11K
     training_epochs: 100
     learning_rate: 0.001
     optimizer: SGD
     num_gpus: 0

   testing: {}
   compilation: {}

Run an example from the tinyml-modelzoo folder:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/radar_posture_and_fall_detection/config.yaml
   ./run_tinyml_modelzoo.sh examples/radar_surface_classification/config.yaml

See :doc:`/getting_started/quickstart` for install options.

Dataset Format
--------------

.. code-block:: text

   my_radar_dataset/
   └── classes/
       ├── class_a/
       │   └── capture1.csv
       └── class_b/
           └── capture1.csv

Each CSV row is one radar frame. Column names must match what the preset expects (``point_x*``,
``track_velocity_*``, ``bin*`` and so on, see :doc:`/RADAR_PROCESSING_PIPELINE`). Non-numeric header columns
such as session, recording, time and track id are dropped automatically. Windows of ``frame_size`` rows are
built with ``Simple Window`` (``stride_size`` controls the overlap). More in :doc:`/byod/audio_radar_format`.

Available Models
----------------

Parameter counts depend on the number of input features (``frame_size`` x features per frame) and classes. The
values below were counted from ``tinyml_modelzoo/models/radar.py``.

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Model Name
     - Parameters
     - Description
   * - ``RF_MLP_3L_11K``
     - 10,982 (130 inputs, 2 classes)
     - 3-layer MLP (BatchNorm + Linear 64, 32, classes). Default for surface classification.
   * - ``RF_MLP_4L_19K``
     - 19,298 (248 inputs, 2 classes); 11,510 at 130 inputs
     - 4-layer MLP (BatchNorm + Linear 64, 32, 16, classes). Flat input, no fixed feature layout.
   * - ``RF_DualStream_PointCloud_19K``
     - 18,941 (248 inputs, frame_size 8, 5 classes)
     - Separate position and velocity streams fused by a small MLP. Default for posture and fall detection;
       velocity features must be at slots 1-3 of each frame.

All three target ``xWRL6432``. For your own architecture see :doc:`/BRING_YOUR_OWN_MODEL`.

Metrics
-------

The training log reports accuracy, F1-score and the confusion matrix like the other classification tasks;
see :doc:`/getting_started/understanding_run_output`.

Tips
----

* The posture example uses ``within_files`` splitting because each capture is a long continuous recording; the
  surface example uses ``amongst_files``. See :doc:`/byod/data_splitting`.
* Quantization is off by default for the radar models. Set ``training.quantization: 1`` to enable it.
* Preset parameters do not need repeating in your YAML; override only what differs.

See Also
--------

* :doc:`/byod/audio_radar_format` - Radar dataset layout
* :doc:`timeseries_classification` - Time series classification
* :doc:`audio_classification` - Audio classification
