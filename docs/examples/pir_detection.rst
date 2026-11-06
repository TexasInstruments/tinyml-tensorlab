=============
PIR Detection
=============

Edge AI solution for PIR (passive infrared) sensor based motion classification on the MSPM0G5187 with NPU
acceleration, also available for the SimpleLink CC13xx/CC27xx/CC35X1 wireless MCUs.

Example folder: ``examples/pir_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/pir_detection>`_)

Overview
--------

Traditional PIR sensors only detect presence or absence of motion and cannot tell its source. Pets and
environmental factors cause false alarms, and manual threshold tuning is unreliable across environments.
This example classifies PIR sensor signals into motion categories, which reduces false positives while keeping
detection sensitivity.

* **Application**: security systems, smart home automation, occupancy sensing
* **Task**: classification (``task_type: pir_detection``), 3 classes
* **Primary device**: MSPM0G5187
* **Model**: ``PIRDetection_model_1_t``
* **Targets of the reference solution**: real-time classification, more than 90% accuracy, low power

System Components
-----------------

* Hardware: `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ with integrated NPU, and the EdgeAI Sensor
  Boosterpack (`TIDA-010997 <https://www.ti.com/tool/TIDA-010997>`_) with a PIR sensor
* Software: Code Composer Studio 12.x or later, `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_ 2.08.00 or later

Available Configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Config file
     - Device
     - Feature extraction preset / epochs
   * - ``config_MSPM0.yaml``
     - MSPM0G5187
     - ``PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D_FixedPoint``, 100 epochs
   * - ``config_CC1312.yaml``
     - CC1312
     - ``PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D``, 50 epochs
   * - ``config_CC1314.yaml``
     - CC1314
     - same as CC1312
   * - ``config_CC1352.yaml``
     - CC1352
     - same as CC1312
   * - ``config_CC1354.yaml``
     - CC1354
     - same as CC1312
   * - ``config_CC2755.yaml``
     - CC2755
     - same as CC1312
   * - ``config_CC35X1.yaml``
     - CC35X1
     - same as CC1312

The MSPM0 config uses the fixed-point feature extraction that matches the integer pipeline on the device; the
other configs use the floating point version.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         # MSPM0G5187 (primary)
         ./run_tinyml_modelzoo.sh examples/pir_detection/config_MSPM0.yaml
         # other devices
         ./run_tinyml_modelzoo.sh examples/pir_detection/config_CC1352.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\pir_detection\config_MSPM0.yaml

Dataset
-------

``pir_detection_classification`` (``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/pir_detection_classification_dsk.zip``)
contains PIR recordings captured with the EdgeAI Sensor Boosterpack from a range of 6 to 8 meters:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - Human motion
     - Movement patterns characteristic of humans
   * - Background motion
     - Environmental disturbances and noise
   * - Dog motion
     - Movement patterns characteristic of dogs

Feature Extraction Pipeline
---------------------------

The ``PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D`` presets process the raw signal as follows:

1. ADC sampling
2. DC offset removal
3. Windowed processing
4. Symmetric mirroring
5. FFT computation
6. Magnitude calculation
7. Average pooling
8. Additional features: zero crossing rate (ZCR), slope changes, dominant frequency
9. Concatenation of all features into the model input

Configuration (``config_MSPM0.yaml``)
-------------------------------------

.. code-block:: yaml

   common:
     task_type: pir_detection
     target_device: MSPM0G5187
   dataset:
     dataset_name: pir_detection_classification
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/pir_detection_classification_dsk.zip
   data_processing_feature_extraction:
     data_proc_transforms: []
     feature_extraction_name: PIRDetection_125Input_25Feature_25Frame_1InputChannel_2D_FixedPoint
     variables: 1
   training:
     model_name: PIRDetection_model_1_t
     batch_size: 64
     training_epochs: 100

``PIRDetection_model_1_t`` is a compact CNN with about 53K or more parameters, NPU compatible and optimized for
multivariate input signals. The MSPM0 config trains with batch size 64, learning rate 0.00001, weight decay 1e-20 and
100 epochs; the CC13xx/CC27xx/CC35X1 configs use batch size 64, learning rate 0.001 and 50 epochs. The optimizer
is the default SGD. Compilation uses the TI Neural Network Compiler and generates
``model.a`` and ``tvmgen_default.h``.

Results
-------

The example ``readme.md`` reports about 98% accuracy for CC1352 with floating point feature extraction and
about 92.46% for MSPM0 with fixed point feature extraction (earlier release). In the automated 1.5.0 config test
run:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Config
     - Test accuracy
     - AUC ROC
   * - ``config_CC1312``, ``CC1314``, ``CC1352``, ``CC1354``, ``CC2755``, ``CC35X1``
     - 94.81%
     - 0.826
   * - ``config_MSPM0.yaml``
     - 65.40%
     - 0.795

.. note::

   The MSPM0 test-run accuracy (65.40%) is far below the 92.46% the example readme quotes from an earlier release.
   The reasons were not investigated here; check the printed confusion matrix of your own run and consider tuning epochs and
   learning rate for the fixed-point configuration.

References
----------

* `EdgeAI Sensor Boosterpack TIDA-010997 <https://www.ti.com/tool/TIDA-010997>`_
* `TI Neural Network Compiler guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_
