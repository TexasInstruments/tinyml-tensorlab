======================================
MOSFET Junction Temperature Prediction
======================================

Predict the case temperature of a high-voltage switch (MOSFET, IGBT) from its power loss and the reading of a
nearby NTC temperature sensor, using a linear model plus a small AI model for the residual error.

Example folder: ``examples/mosfet_temp_prediction``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/mosfet_temp_prediction>`_)

Overview
--------

* **Task**: timeseries regression (``generic_timeseries_regression``)
* **Application**: power conversion and motor drives, protection of switching devices without a direct temperature sensor
* **Device**: F29H85 (``config.yaml``)
* **Model**: ``REGR_3k`` (an MLP-style regression model)
* **Target**: ``T_case - T_linear``, the residual between the measured case temperature and the linear model output

Thermal Model
-------------

The power loss of the switch (conduction plus switching loss, summed over all devices in a module) raises the
junction temperature, which propagates to the case and then through the heat sink to the ambient. An NTC sensor
next to the device observes the temperature, but its reading can differ significantly from the case temperature.

.. figure:: /_static/img/examples/mosfet_temp/thermal_model.png
   :width: 720px
   :align: center
   :alt: Simplified thermal model of a high voltage switch

   Simplified thermal model of a high-voltage switch

The thermal model from ``T_junction`` to ``T_case`` is normally provided by the device manufacturer. The example
focuses on predicting ``T_case`` from ``P_loss`` and ``T_NTC``. The coupling between ``T_case`` and ``T_NTC``
depends on the mechanical and cooling design and varies with coolant temperature, coolant flow rate and ambient
temperature.

Linear + AI Model
-----------------

.. figure:: /_static/img/examples/mosfet_temp/temp_model.png
   :width: 500px
   :align: center
   :alt: Linear plus AI temperature model

   Linear plus AI model

The prediction has two components:

* **Linear model**: a linear autoregressive-moving-average (ARMA) model of the linear thermal behavior from
  ``P_loss`` to the difference between ``T_case`` and ``T_NTC``.
* **AI model**: an MLP that models the residual between the linear prediction and the true measurement. This
  residual can be highly nonlinear in ``P_loss`` and ``T_NTC``.

Both models can take extra inputs such as coolant flow rate, coolant temperature and ambient temperature. The
combination is more accurate because the AI model captures nonlinear behavior, more constrained because the linear
model captures most of the behavior, and more robust overall.

The Dataset
-----------

The edge AI model does not predict the case temperature directly; it predicts the difference between the
expected temperature and the linear model output.

.. figure:: /_static/img/examples/mosfet_temp/input_buffer.png
   :width: 720px
   :align: center
   :alt: Input buffer layout

   Input buffer for one prediction

For every predicted value the data holds 20 past NTC temperature points, 20 past power points and 3 fixed
meta-information values (ambient temperature, coolant temperature and coolant flow rate, constant within a
file). That is 43 input columns, which is why the config lists ``variables`` 1 to 43 with ``frame_size: 1``. The
data covers several ambient temperatures, coolant temperatures and coolant flow rates. The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/mosfet_temp_prediction.zip``.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/mosfet_temp_prediction/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\mosfet_temp_prediction\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F29H85
   dataset:
     dataset_name: mosfet_temp_prediction
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/mosfet_temp_prediction.zip
   data_processing_feature_extraction:
     data_proc_transforms: ['SimpleWindow']
     variables: [1,2,3, ... ,43]      # the config lists every column index from 1 to 43
     stride_size: 1
     frame_size: 1
   training:
     model_name: REGR_3k
     batch_size: 128
     training_epochs: 100
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     learning_rate: 0.01
     quantization: 0
     output_int: false
   testing: {}
   compilation: {}

``quantization: 0`` trains a float model. Other regression models (``REGR_*``) can be selected; see
:doc:`/appendix/model_zoo_reference`.

Results
-------

The example readme does not publish accuracy numbers. In the automated 1.5.0 config test run (early stopped)
the float model reached a best-epoch MSE of 0.084 and R2 of 0.962 on validation data, and a test-set RMSE of
0.37 with R2 of 0.95.

Output Location
---------------

Results are written to ``./data/projects/mosfet_temp_prediction/run/<date-time>/REGR_3k/``. The device files are
``compilation/artifacts/mod.a``, ``compilation/artifacts/tvmgen_default.h`` and the golden vectors in
``training/quantization/golden_vectors/``.

Running on Device
-----------------

Use the ``generic_timeseries_regression`` example of the F29H85x SDK and follow
:doc:`/deploying_regression_models_from_modelmaker_to_device/readme` and :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

See Also
--------

* :doc:`torque_measurement_regression`
* :doc:`induction_motor_speed_prediction`
