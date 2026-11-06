=============
First Example
=============

A detailed walkthrough of DC arc fault detection, from config to compiled model.

Overview
--------

DC arc faults are dangerous electrical arcs in DC systems (solar, batteries).
The model classifies windows of current samples as ``class_0_normal`` or ``class_1_arc``.
The outputs shown below are from a real run of this example on the F28P55 with
``CLS_1k_NPU``.

Step 1: The Configuration
-------------------------

``examples/dc_arc_fault/config_dsk.yaml`` (comments omitted):

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
     batch_size: 32
     training_epochs: 10
     num_gpus: 0
     early_stopping: true
     early_stopping_patience: 5
   testing: {}
   compilation: {}

``common``
   ``task_type: arc_fault`` selects the arc fault classification task and
   ``target_device: F28P55`` is an NPU device.

``dataset``
   ``dataset_name: dc_arc_fault_example_dsk`` is a built-in dataset preset that is
   downloaded and extracted automatically.

``data_processing_feature_extraction``
   A 1024-point FFT per frame with 256 features, one frame, one variable (current).

``training``
   The 1K-parameter NPU classification model. Training stops early when validation
   accuracy has not improved for 5 epochs.

Step 2: Run
-----------

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsk.yaml

   .. tab:: Windows (cmd)

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\dc_arc_fault\config_dsk.yaml

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         ./run_tinyml_modelzoo.ps1 examples/dc_arc_fault/config_dsk.yaml

The console shows, in order: dataset download, feature extraction, float training,
quantization-aware training, testing, and compilation.

Step 3: Read the Training Log
-----------------------------

The log reports validation accuracy per epoch and a summary of the best epoch, with
accuracy, F1 score, AUC ROC and the confusion matrix. When early stopping triggers,
the log says so (``Early stopping at epoch N: no improvement for 5 epochs``) and the
best checkpoint is kept. The test-set result is printed as ``Test Data Evaluation Accuracy``.

Step 4: Examine the Output Folder
---------------------------------

The output goes to ``./data/projects/dc_arc_fault_example_dsk/run/<date-time>/CLS_1k_NPU/``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Path
     - Contents
   * - ``training/base/``
     - Float model (``model.onnx``, ``checkpoint.pth``), ``run.log``, PCA plots, golden vectors
   * - ``training/quantization/``
     - Quantized model, golden vectors, ``post_training_analysis/`` (ROC curve, score histogram, ``fpr_tpr_thresholds.csv``)
   * - ``compilation/artifacts/``
     - ``mod.a`` (compiled model library) and ``tvmgen_default.h`` (C API)
   * - ``compilation/<date-time>_CLS_1k_NPU_F28P55.zip``
     - Bundle with everything a device project needs
   * - ``compilation/run.log``
     - Compiler log with the memory usage of the model library

Every item is explained in :doc:`understanding_run_output`.

Step 5: Customize
-----------------

.. code-block:: yaml

   training:
     model_name: CLS_4k_NPU       # a larger model (check the model name exists in the model zoo)

.. code-block:: yaml

   common:
     target_device: F2837         # a non-NPU C2000 device

Use your own data by changing the ``dataset`` section (see :doc:`/CONFIG_YAML_GUIDE` and :doc:`/byod/index`).
Model names available for your task and device are listed under :doc:`/appendix/model_zoo_reference`.

Next Steps
----------

* :doc:`understanding_run_output`
* :doc:`/DEPLOYING_TO_SDK_EXAMPLE`
* :doc:`/examples/index`
