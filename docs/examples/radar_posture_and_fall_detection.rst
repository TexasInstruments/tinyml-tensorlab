==================================
Radar Posture and Fall Detection
==================================

Classify the posture of a person (falling, laying, sitting, standing, walking) from mmWave radar point cloud and
track data on the xWRL6432. Introduced in 1.5.0 as ``radar_pose_and_fall_detection``; renamed, reworked and
documented in detail in 1.6.0.

Example folder: ``examples/radar_posture_and_fall_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/radar_posture_and_fall_detection>`_).
The folder has its own detailed README, ``config.yaml`` and ``annotate_radar_dataset.py``.

Overview
--------

* **Task**: ``radar_point_cloud_classification`` (``target_module: radar``), 5 classes
* **Application**: human posture recognition and fall detection from radar point clouds, for example for elder
  care and in-home monitoring
* **Device**: xWRL6432 (Arm Cortex-M4F, no NPU; compiled with ``ti-cgt-armllvm``)
* **Feature extraction**: preset ``Generic_248Input_Posture_and_Fall_Detection``: 8 frames x 31 features per
  frame = 248 inputs (track height, velocity x/y/z, acceleration x/y/z and 8 highest/lowest points of
  ``y``, ``z``, ``snr``)
* **Model**: ``RF_DualStream_PointCloud_19K`` (about 18.9K parameters at 248 inputs): a position stream, a
  velocity stream and a fusion MLP. Velocity features must stay at slots 1-3 of every frame, which the preset
  guarantees. ``RF_MLP_4L_19K`` and ``RF_MLP_3L_11K`` accept any feature order.

The Dataset
-----------

The config downloads ``posture_and_fall_detection_dataset.zip`` from
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/``. If the download fails on a corporate network, download
it manually and set ``dataset.input_data_path`` to the extracted folder. Data is organised in class folders
(``classes/``), one folder per posture:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Index
     - Class
   * - 0
     - Falling
   * - 1
     - Laying
   * - 2
     - Sitting
   * - 3
     - Standing
   * - 4
     - Walking

Column names must match the radar signal chain output (``point_x*``, ``track_velocity_*`` and so on). Use
``annotate_radar_dataset.py`` in the example folder to annotate your own recordings, and
:doc:`/RADAR_PROCESSING_PIPELINE` for the column reference and every pre-processing parameter.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/radar_posture_and_fall_detection/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\radar_posture_and_fall_detection\config.yaml

Configuration
-------------

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

* ``split_type: within_files`` splits every file into 70% train, 20% validation and 10% test windows.
* The preset sets ``frame_size: 8`` and ``stride_size: 0.125``; override them only if your use case differs.
* Quantization is off by default; set ``training.quantization: 1`` to enable it.

Output Location
---------------

Results are written to ``./data/projects/radar_point_cloud_classification/run/<date-time>_RF_DualStream_PointCloud_19K/``.
The trained model is in ``training/base`` and the compiled model in ``compilation``. See
:doc:`/getting_started/understanding_run_output` and :doc:`/task_types/radar_classification`.

See Also
--------

* :doc:`/task_types/radar_classification`
* :doc:`radar_surface_classification`
* :doc:`dynamic_hand_gesture_recognition`
* :doc:`fall_detection_classification`
