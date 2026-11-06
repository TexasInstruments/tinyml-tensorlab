=============================
Radar Surface Classification
=============================

Classify the surface or material in front of a TI mmWave radar (for example grass vs not grass) directly from
1-D FFT range-profile data on the xWRL6432. No point cloud or tracker is needed: the shape of the energy return
across range bins is the feature. New in 1.6.0.

Example folder: ``examples/radar_surface_classification``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/radar_surface_classification>`_).
The folder has its own detailed README and ``config.yaml``.

Overview
--------

* **Task**: ``radar_surface_classification`` (``target_module: radar``)
* **Device**: xWRL6432 (Arm Cortex-M4F, no NPU)
* **Dataset**: ``radar_grass_surface_dataset.zip`` (grass vs not grass), downloaded by the config
* **Feature extraction**: preset ``Generic_130Input_Radar_SurfaceClassification``: range bins 6-31 (26 bins) x
  5 frames = 130 inputs, ``stride_size: 0.2``
* **Model**: ``RF_MLP_3L_11K`` (10,982 parameters at 130 inputs, 2 classes). ``RF_MLP_4L_19K`` also works and
  adapts its first layer to the input size (11,510 parameters at 130 inputs).

Running the Example
-------------------

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/radar_surface_classification/config.yaml

Configuration
-------------

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
     radar_range_bins_bounds: [null, null]
   training:
     model_name: RF_MLP_3L_11K
     training_epochs: 100
     learning_rate: 0.001
     optimizer: SGD
     output_int: false
     num_gpus: 0
   testing: {}
   compilation: {}

* ``radar_range_bins_bounds: [null, null]`` keeps all range bins; the preset default is ``[6, 31]``. Changing
  the bins or ``frame_size`` changes the number of model inputs.
* Input CSV columns are ``bin0`` ... ``binN``, one row per frame. See :doc:`/RADAR_PROCESSING_PIPELINE`.
* ``split_type: amongst_files`` keeps whole recordings in one split.
* On the device, the same range-bin selection and windowing must be applied to the live FFT output before
  calling ``tvmgen_default_run()``.

See Also
--------

* :doc:`/task_types/radar_classification`
* :doc:`radar_posture_and_fall_detection`
