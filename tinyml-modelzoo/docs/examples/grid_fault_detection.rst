====================
Grid Fault Detection
====================

Detect AC grid faults for single-phase on-board chargers (OBC) of electric vehicles, running on an F29x MCU.

Example folder: ``examples/grid_fault_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/grid_fault_detection>`_)

Overview
--------

Nearly all EVs and PHEVs have an on-board charger that must be robust to faults and abnormal conditions of the
AC grid. Grid faults have highly variable signatures and are hard to detect with traditional threshold based
heuristics. An edge-AI model running on the same MCU that controls the OBC power stage can protect the OBC
against adverse grid events, log abnormal events, and potentially save the OBC.

TI's approach uses a CNN trained on a proprietary grid fault dataset and running on the F29x MCU.

.. figure:: /_static/img/examples/grid_fault_detection/grid_fault_variation.png
   :width: 500px
   :align: center
   :alt: Variation of AC grid fault waveforms

   Grid fault waveforms vary strongly between events

* **Application**: EV on-board charger protection, grid event logging, power-stage safety
* **Task**: classification (``generic_timeseries_classification``)
* **Data**: univariate AC grid current, 16 samples per file
* **Device**: F29H85
* **Model**: ``CLS_1k_NPU``

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/grid_fault_detection/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\grid_fault_detection\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F29H85
   dataset:
     dataset_name: grid_fault_detection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/grid_fault_dataset.zip
   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 16
     stride_size: 1
     variables: 1
   training:
     model_name: CLS_1k_NPU
     num_gpus: 0
     batch_size: 512
     training_epochs: 250
     early_stopping: true
     early_stopping_patience: 5
   testing: {}
   compilation: {}

Dataset and Feature Extraction
------------------------------

There are no well-defined fault categories for AC grid faults. The categories were defined with a hybrid
annotation technique that combines human annotation and unsupervised annotation using hierarchical density
based clustering; the quality of the annotation was evaluated with dimensionality reduction and manual QC.

For this example the feature extraction presets of ModelZoo are bypassed: ``SimpleWindow`` only frames the
pre-processed data. The dataset has only 16 values per file, so ``frame_size`` must not exceed 16.

Results
-------

The example readme does not publish accuracy numbers. In the automated 1.5.0 config test run the config finished
quickly (early stopping is enabled with patience 5) with a test accuracy of 47.31% and an AUC ROC of 0.563, so
the shipped configuration should be treated as a starting point that needs tuning (for example a longer
patience or a different model) rather than a validated accuracy reference.

Deployment
----------

See :doc:`/deploying_classification_models_from_modelmaker_to_device/readme` and
:doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

See Also
--------

* :doc:`electrical_fault`: transmission line fault classification
* :doc:`grid_stability`: power grid stability prediction
