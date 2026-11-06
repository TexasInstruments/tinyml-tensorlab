==============================
Torque Measurement Regression
==============================

Estimate the electromagnetic torque of a permanent magnet synchronous motor (PMSM) from electrical and
thermal measurements, as a virtual (sensorless) torque sensor.

Example folder: ``examples/torque_measurement_regression``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/torque_measurement_regression>`_)

Overview
--------

* **Task**: timeseries regression (``generic_timeseries_regression``)
* **Application**: motor control, sensorless torque estimation
* **Device**: F28P55 (``config.yaml``)
* **Model**: ``REGR_10k``
* **Target**: ``torque``

The Dataset
-----------

The data are measurements from a PMSM prototype on a test bench (German OEM motor, recorded by the LEA
department at Paderborn University), sampled at 2 Hz. It is the
`Electric Motor Temperature dataset on Kaggle <https://www.kaggle.com/wkirgsn/electric-motor-temperature>`_.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``i_d``, ``i_q``
     - Current d- and q-components
   * - ``u_d``, ``u_q``
     - Voltage d- and q-components
   * - ``motor_speed``
     - Motor speed
   * - ``ambient``, ``coolant``
     - Ambient and coolant temperature
   * - ``pm``
     - Permanent magnet surface temperature
   * - ``stator_winding``, ``stator_tooth``, ``stator_yoke``
     - Stator temperatures
   * - ``torque``
     - Torque induced by current (the target)

The example uses the first 10 columns (``i_d`` to ``stator_tooth``) as inputs (``variables: 10``) and predicts
``torque``. The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/torque_measurement_dataset.zip``. The dataset follows
the regression layout described in :doc:`/byod/regression_format`.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/torque_measurement_regression/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\torque_measurement_regression\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F28P55
   dataset:
     dataset_name: torque_measurement
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/torque_measurement_dataset.zip
   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     variables: 10
     frame_size: 128
     stride_size: 0.25
   training:
     model_name: REGR_10k
     batch_size: 512
     training_epochs: 50
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 1
     learning_rate: 0.001
     output_int: false
     auto_quantization: true
   testing: {}
   compilation: {}

``num_gpus: 1`` expects a GPU; set it to 0 to train on CPU. Try other ``frame_size`` and ``stride_size``
values to see how they affect training. In the readme experiment the smaller frame (128) gave the better
R2 score.

Results
-------

The example readme reports these results for two frame sizes (stride size 0.25):

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Frame size
     - Float R2 (MSE)
     - Quantized R2 (MSE)
     - Cycles
   * - 128
     - 0.994 (12.98)
     - 0.963 (83.59)
     - 114102
   * - 256
     - 0.983 (36.82)
     - 0.948 (113.36)
     - 187445

For frame size 128 the readme also reports a test-set RMSE of 9.40, MSE of 88.36 and R2 of 0.98. The cycle
counts are taken from the readme and were measured on the device for the quantized model.

Output Location
---------------

Results are written to ``./data/projects/torque_measurement/run/<date-time>/REGR_10k/``. The four files needed
on the device are ``compilation/artifacts/mod.a``, ``compilation/artifacts/tvmgen_default.h``,
``training/quantization/golden_vectors/test_vector.c`` and
``training/quantization/golden_vectors/user_input_config.h``. See
:doc:`/getting_started/understanding_run_output`.

Running on Device
-----------------

The C2000Ware (or tinyml-sdk) example ``f28p55x_generic_timeseries_regression`` takes these four files. Follow
:doc:`/deploying_regression_models_from_modelmaker_to_device/readme` and :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.
Check the ``test_result`` variable for the cases in ``test_vector.c``.

See Also
--------

* :doc:`generic_regression`
* :doc:`induction_motor_speed_prediction`
* :doc:`mosfet_temp_prediction`
