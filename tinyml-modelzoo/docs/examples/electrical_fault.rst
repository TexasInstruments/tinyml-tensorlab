================
Electrical Fault
================

Detect and classify transmission line faults from three-phase voltage and current measurements, with a
focus on solving the multicollinearity problem of raw voltage and current channels through frequency
domain feature extraction.

Example folder: ``examples/electrical_fault``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/electrical_fault>`_)

Overview
--------

The dataset is a multivariate time series obtained by simulating a three-phase transmission line in MATLAB
Simulink. The six variables are the phase voltages (Va, Vb, Vc) and phase currents (Ia, Ib, Ic).

* **Application**: power grid protection, substation automation, fault localization
* **Task**: time series classification (``generic_timeseries_classification``)
* **Variants**: 2-class fault detection (fault / no fault) and 6-class fault type classification

.. figure:: /_static/img/examples/electrical_fault/simulink.png
   :width: 700px
   :align: center
   :alt: Simulink model for the electrical fault dataset

   Simulink model used to generate the electrical fault data

Dataset Variants
----------------

The raw data is available as ``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/electrical_fault_raw.zip``.
Two helper scripts in the example folder convert it to ModelZoo datasets.

.. list-table::
   :header-rows: 1
   :widths: 22 28 22 28

   * - Script
     - Output dataset
     - Classes
     - Use case
   * - ``electrical_fault.py``
     - ``electrical_fault_dataset.zip``
     - 2 (fault / no fault)
     - Binary fault detection
   * - ``electrical_fault_6class.py``
     - ``electrical_fault_6class_dataset.zip``
     - 6 (from the G, C, B, A target flags)
     - Fault type classification

The shipped configs already point to pre-built copies of these zip files on the TI download server, so
running the scripts is only needed if you want to regenerate the datasets locally:

.. code-block:: bash

   cd examples/electrical_fault
   python electrical_fault.py          # creates electrical_fault_dataset.zip
   python electrical_fault_6class.py   # creates electrical_fault_6class_dataset.zip

Then set ``dataset.input_data_path`` to the generated zip, for example
``examples/electrical_fault/electrical_fault_dataset.zip``.

In the 6-class data the target flags [G, C, B, A] encode the fault type, for example [0, 0, 0, 0] is no fault,
[0, 1, 1, 0] is a line-to-line fault between nodes B and C, [1, 0, 0, 1] is a line-to-ground fault at node A,
and [1, 1, 1, 1] is a fault involving all nodes and ground.

Available Configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 14 22 40

   * - Config file
     - Device
     - Model
     - Notes
   * - ``config.yaml``
     - F28P55
     - ``CLS_ResCat_3k``
     - 2-class dataset ``electrical_fault_2class``, 6 variables, 256-sample frames, FFT features (128 per frame)
   * - ``config_MSPM0.yaml``
     - MSPM0G5187
     - ``ElectricalFault_model_40k_t``
     - 6-class dataset, 3 variables, 128-sample frames, Q15 FFT features (32 per frame, 8 frames), 25 epochs

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         # 2-class fault detection on F28P55
         ./run_tinyml_modelzoo.sh examples/electrical_fault/config.yaml

         # 6-class fault type classification on MSPM0G5187
         ./run_tinyml_modelzoo.sh examples/electrical_fault/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\electrical_fault\config.yaml
         run_tinyml_modelzoo.bat examples\electrical_fault\config_MSPM0.yaml

Configuration (``config.yaml``)
-------------------------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: electrical_fault_2class
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/electrical_fault_dataset.zip
   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     stride_size: 0.01
     variables: 6
     feature_extraction_name: Custom_Default
     feat_ext_transform: ['FFT_FE', 'FFT_POS_HALF', 'DC_REMOVE', 'ABS', 'LOG_DB', 'CONCAT']
     frame_size: 256
     feature_size_per_frame: 128
     num_frame_concat: 1
   training:
     model_name: CLS_ResCat_3k
     num_gpus: 0
     early_stopping: true
     early_stopping_patience: 5
   testing: {}
   compilation: {}

Multicollinearity and Feature Extraction
----------------------------------------

Raw phase voltages and currents are strongly correlated (the example readme shows correlation
coefficients of about -0.94 to -0.95 between pairs such as Ia and Vc, or Ib and Va). Training directly on such
channels can fail to find good hyperparameters. Two remedies are to remove highly correlated features, or to
extract features that reduce dimensionality. This example uses the second one: an FFT of each frame
(``FFT_FE``, ``FFT_POS_HALF``, ``DC_REMOVE``, ``ABS``), an optional ``BINNING`` step, ``LOG_DB`` and
``CONCAT``. Spectral features decorrelate the channels because they describe how energy is distributed over
frequencies. The example ``readme.md`` walks through the transform options step by step. You can try disabling
the features to see the effect:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: Custom_Default
     feat_ext_transform: []

Results from the 1.5.0 Test Run
-------------------------------

The example readme does not publish accuracy numbers. From the automated 1.5.0 config test run:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Config
     - Test accuracy
     - AUC ROC
   * - ``config.yaml`` (2-class, F28P55)
     - 99.44%
     - 1.000
   * - ``config_MSPM0.yaml`` (6-class, MSPM0G5187)
     - 88.89%
     - 0.981

See Also
--------

* :doc:`grid_stability`: power grid stability prediction
* :doc:`grid_fault_detection`
* Using your own data: :doc:`/byod/classification_format`
