================================
Induction Motor Speed Prediction
================================

Predict the speed (RPM) of a three-phase induction motor from supply, motor and operating parameters.

Example folder: ``examples/induction_motor_speed_prediction``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/induction_motor_speed_prediction>`_)

Overview
--------

* **Task**: timeseries regression (``generic_timeseries_regression``)
* **Application**: motor monitoring and sensorless speed estimation
* **Device**: F29H85 (``config.yaml``)
* **Model**: ``REGR_1k`` (two BatchNorm + Conv + ReLU layers and two light linear layers)
* **Target**: motor speed in RPM

The Dataset
-----------

The dataset is simulated: 15,000 observations of three-phase induction motors under varied electrical and
mechanical conditions. It combines physics-based relations (frequency determines synchronous speed) with
non-linear interactions (load, voltage and temperature affect slip).

* **Supply**: voltage (200 to 460 V), current, frequency (50 or 60 Hz), power factor (0.65 to 0.95)
* **Motor**: number of poles (2, 4, 6, 8), stator resistance
* **Operating conditions**: load torque, ambient temperature
* **Target**: motor speed (RPM), derived from synchronous speed and slip

.. figure:: /_static/img/examples/induction_motor/induction_motor_relationships.png
   :width: 720px
   :align: center
   :alt: Electromechanical relationships of the induction motor dataset

   Electromechanical relationships used to generate the dataset

The key relations are ``Sync = 120 * Frequency / Poles``, ``Slip = (Sync - Actual) / Sync`` and
``Actual = Sync * (1 - Slip)``. Active power ``P = V * I * PF`` and apparent power ``S = V * I`` are also
included.

Preparing the Dataset
---------------------

``config.yaml`` downloads ``induction_motor_speed_prediction_dataset.zip`` from
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/``. To regenerate the zip locally, run the helper
script and point ``dataset.input_data_path`` at the output:

.. code-block:: bash

   cd examples/induction_motor_speed_prediction
   python induction_motor_speed_prediction.py

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/induction_motor_speed_prediction/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\induction_motor_speed_prediction\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F29H85
   dataset:
     dataset_name: induction_motor_speed_prediction
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/induction_motor_speed_prediction_dataset.zip
   data_processing_feature_extraction:
     frame_size: 8
     data_proc_transforms: ['SimpleWindow']
     stride_size: 1
     feature_extraction_name: Generic_8Input_ABS_8Feature_1Frame
     variables: 11
   training:
     model_name: REGR_1k
     batch_size: 256
     training_epochs: 50
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     learning_rate: 0.000001
     output_int: false
   testing: {}
   compilation: {}

The dataset is not a true timeseries, so a small frame size works better. The shipped config uses
``frame_size: 8`` together with the ``Generic_8Input_ABS_8Feature_1Frame`` feature extraction preset
(absolute value of 8 inputs). The readme also reports frame sizes 1 and 4; these need a ``frame_size`` and
feature extraction setting that match each other.

Results
-------

The example readme reports results for three frame sizes:

.. list-table::
   :header-rows: 1
   :widths: 16 28 28 28

   * - Frame size
     - Float R2 (MSE)
     - Quantized R2 (MSE)
     - Exported model R2 (RMSE)
   * - 1
     - 1.000 (185.94)
     - 0.999 (451.44)
     - 0.990 (87.46)
   * - 4
     - 0.997 (645.68)
     - 0.997 (785.67)
     - 0.990 (46.36)
   * - 8
     - 0.978 (2347.63)
     - 0.978 (2341.76)
     - 0.970 (59.88)

Device measurements from the readme (F29H85x, flash and SRAM from the map file, cycles from the profiling API):

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Frame size
     - Flash (B)
     - SRAM (B)
     - Cycles
     - Inference time (us)
   * - 1
     - 4306
     - 128
     - 1889
     - 9.45
   * - 4
     - 4582
     - 130
     - 3519
     - 17.59
   * - 8
     - 5292
     - 256
     - 7288
     - 36.44

These numbers come from the example readme (written for 1.3.0) and were not regenerated for 1.5.0.

Output Location
---------------

Results are written to ``./data/projects/induction_motor_speed_prediction/run/<date-time>/REGR_1k/``. The files
for the device are ``compilation/artifacts/mod.a``, ``compilation/artifacts/tvmgen_default.h``,
``training/quantization/golden_vectors/test_vector.c`` and
``training/quantization/golden_vectors/user_input_config.h``.

Running on Device
-----------------

Copy the four files into the ``generic_timeseries_regression`` example of the F29H85x SDK
(``examples/ai/generic_timeseries_regression/f29h85x``), build, flash and check the ``test_result`` variable.
See :doc:`/deploying_regression_models_from_modelmaker_to_device/readme` and :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

See Also
--------

* :doc:`generic_regression`
* :doc:`torque_measurement_regression`
