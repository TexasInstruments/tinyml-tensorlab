=======================
Gearbox Fault Detection
=======================

Classify gearbox operating conditions (healthy or broken tooth) from 4-channel vibration data on the
MSPM0G5187 MCU with integrated NPU.

Example folder: ``examples/gearbox_fault_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/gearbox_fault_detection>`_)

Overview
--------

Gearbox failures are a leading cause of unplanned downtime in industrial machinery, and traditional condition
monitoring needs manual inspection. This example shows real-time fault diagnosis directly on a resource-constrained
MCU without cloud connectivity.

* **Application**: predictive maintenance, industrial machinery, gearbox health monitoring
* **Task**: binary classification (``generic_timeseries_classification``)
* **Data**: 4 vibration channels (``variables: 4``), 256-sample windows
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config)
* **Models**: ``GearboxFault_model_1.2k_t`` (default), ``GearboxFault_model_1.5k_t``

System Components
-----------------

* Hardware: `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ with integrated NPU
* Software: Code Composer Studio 12.x or later, MSPM0 SDK 2.10.00 or later

Dataset
-------

The example uses the SpectraQuest Gearbox Fault Diagnostics Simulator data from
`Kaggle <https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis>`_, packaged as
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/gearbox_dataset.zip`` (downloaded automatically).

* **Classes**: 2 (healthy, broken tooth)
* **Operating conditions**: 10 load levels (0 to 90%)
* **Columns**: ``a1``, ``a2``, ``a3``, ``a4`` (four accelerometers in four directions), about 88,000 rows per file
* **File naming**: ``{b|h}30hz{load}.csv`` where ``b`` is broken tooth and ``h`` is healthy

Feature Extraction
------------------

No frequency transform is used: the 4-channel accelerometer data is cut into fixed 256-sample windows
(``SimpleWindow``, ``frame_size: 256``, ``stride_size: 1``) and passed as a single frame to the model with
``stacking: 2D1``.

Models
------

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Model
     - Parameters
     - Flash (bytes)
     - Description
   * - ``GearboxFault_model_1.2k_t``
     - about 1,174
     - about 8,233
     - 4-layer network, progressive channel reduction (12, 12, 8, 8)
   * - ``GearboxFault_model_1.5k_t``
     - about 1,914
     - about 8,878
     - 3-layer network, constant channels (16, 16, 16)

Both models use INT8 quantization with a 4-channel, 256-sample input. The parameter and flash figures are from
the example readme.

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: gearbox_fault_detection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/gearbox_dataset.zip
   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     frame_size: 256
     stride_size: 1
     variables: 4
     stacking: '2D1'
   training:
     model_name: GearboxFault_model_1.2k_t   # or GearboxFault_model_1.5k_t
     batch_size: 256
     training_epochs: 50
     num_gpus: 0
   testing: {}
   compilation: {}

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/gearbox_fault_detection/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\gearbox_fault_detection\config_MSPM0.yaml

Results
-------

The example readme states an accuracy of 97 to 100% depending on the model. In the automated 1.5.0 config test
run, ``config_MSPM0.yaml`` reached a test accuracy of 100.00% with an AUC ROC of 1.000.

See Also
--------

* :doc:`motor_bearing_fault`: bearing fault classification from vibration data
* :doc:`fan_blade_fault_classification`: fan blade fault classification
* `TI Neural Network Compiler guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
