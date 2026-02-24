====================
Grid Fault Detection
====================

Detect AC grid faults in on-board chargers for electric vehicles.

Overview
--------

This example demonstrates single-phase grid fault detection for on-board
chargers (OBCs) used in EVs/PHEVs. The on-board charger is expected to be
robust to fault or abnormal conditions of the AC grid. Due to the nature of
the AC grid, faults may have highly variable signatures and are not easy to
detect with traditional threshold-based heuristic criteria. Using an edge-AI
model running on the same MCU that controls the OBC power-stage, it is
possible to protect the OBC against adverse grid events, log abnormal events,
and potentially save the OBC.

TI's approach leverages a Convolutional Neural Network (CNN) edge-AI model
trained on a proprietary grid-fault dataset, running on the F29x MCU. This
enables more accurate and reliable grid fault detection in on-board charging
applications.

**Application**: EV on-board charger protection, grid event logging, power-stage safety

**Task Type**: Time Series Classification

**Data Type**: Univariate (AC grid current)

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F29H85'

   dataset:
     dataset_name: 'grid_fault_detection'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/grid_fault_dataset.zip'

   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 16
     stride_size: 1
     variables: 1

   training:
     model_name: 'CLS_1k_NPU'
     batch_size: 512
     training_epochs: 250

   testing: {}
   compilation: {}

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/grid_fault_detection/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\grid_fault_detection\config.yaml

Dataset Details
---------------

**Input Variables**:

* AC grid current (1 channel, 16 samples per window)

**Feature Extraction**:

Given that there are no well-defined fault categories for AC grid faults, a
hybrid dataset annotation technique is used that leverages a combination of
human annotation and unsupervised annotation using hierarchical density-based
clustering. The goodness of annotation is evaluated using dimensionality
reduction techniques with manual QC. Feature extraction is handled externally
for this example, with ``SimpleWindow`` used to frame the pre-processed data.

**Dataset Download**:

The dataset is automatically downloaded from:

.. code-block:: yaml

   dataset:
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/grid_fault_dataset.zip'

See Also
--------

* :doc:`electrical_fault` - Transmission line fault classification
* :doc:`grid_stability` - Power grid stability prediction
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/grid_fault_detection>`_
