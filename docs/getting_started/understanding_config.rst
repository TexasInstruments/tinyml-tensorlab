====================
Understanding Config
====================

Every run is driven by one YAML file with up to six sections. Keys you leave out use their
defaults, so ``testing: {}`` and ``compilation: {}`` are valid as written.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Purpose
   * - ``common``
     - ``task_type``, ``target_device``; optionally ``projects_path``, ``run_name``
   * - ``dataset``
     - ``dataset_name`` and where the data comes from (``input_data_path``: URL, ``.zip`` or folder)
   * - ``data_processing_feature_extraction``
     - Windowing and feature extraction: a preset (``feature_extraction_name``) or custom transforms
   * - ``training``
     - ``model_name``, ``batch_size``, ``training_epochs``, early stopping, quantization, ``num_gpus``
   * - ``testing``
     - Evaluation of the trained model
   * - ``compilation``
     - Compilation with TI's Neural Network Compiler (``compile_preset_name``)

Minimal Config
--------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: generic_timeseries_classification
   data_processing_feature_extraction:
     feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
     variables: 1
   training:
     model_name: CLS_1k_NPU
     training_epochs: 20
   testing: {}
   compilation: {}

Common Keys
-----------

* ``common.target_device``: the MCU to compile for, e.g. ``F28P55``, ``MSPM0G5187``, ``F29H85``, ``CC2755``, ``AM263``
* ``common.projects_path``: where ``data/projects`` is created (default ``./data/projects``)
* ``training.early_stopping`` and ``training.early_stopping_patience``: stop when validation stops improving
* ``training.num_gpus``: 0 for CPU, 1 or more for CUDA (or MPS on Apple Silicon)
* ``<stage>.enable``: set ``False`` to skip a stage (for example ``compilation: {enable: False}``)

The full walkthrough (4-step recipe, feature extraction presets, custom transforms, useful
knobs and pitfalls) is in the :doc:`/CONFIG_YAML_GUIDE`. There is also an
`interactive slide deck <../slides/config_yaml_guide.html>`_.

Next Steps
----------

* :doc:`/CONFIG_YAML_GUIDE`
* :doc:`running_examples`
* :doc:`/byod/index`
