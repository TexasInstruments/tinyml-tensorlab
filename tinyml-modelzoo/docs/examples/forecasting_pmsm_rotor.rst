==================================
PMSM Rotor Temperature Forecasting
==================================

Forecast the permanent magnet (rotor) temperature of a permanent magnet synchronous motor (PMSM) one step ahead,
without a temperature sensor inside the motor.

Example folder: ``examples/forecasting_pmsm_rotor_temp``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/forecasting_pmsm_rotor_temp>`_)

Overview
--------

* **Task**: timeseries forecasting (``generic_timeseries_forecasting``)
* **Application**: motor protection and derating. Magnets can lose strength permanently above about 150 C, and motor
  life shortens as temperature rises, but sensors deep inside a running motor are costly and fragile.
* **Configs and devices**:

  * ``config.yaml``: F28P55
  * ``config_MSPM0.yaml``: MSPM0G5187

* **Model**: ``FCST_LSTM8`` (an LSTM, 393 parameters according to the readme)
* **Target**: ``pm`` (permanent magnet surface temperature), one step ahead

The Dataset
-----------

The data are measurements from a PMSM prototype on a test bench (German OEM motor, recorded by the LEA department
at Paderborn University), sampled at 2 Hz:
`Electric Motor Temperature (Kaggle) <https://www.kaggle.com/wkirgsn/electric-motor-temperature>`_.

From the original columns the example keeps signals that are easy to measure: ambient temperature (``ambient``),
coolant temperature (``coolant``), the voltage components (``u_d``, ``u_q``) and the phase current magnitude
``i_a = sqrt(i_d^2 + i_q^2)``. Using past values of these and of the permanent magnet temperature ``pm``, the
model forecasts ``pm`` for the next time step. The config uses ``variables: 6`` (``i_a``, ``u_d``, ``u_q``,
``ambient``, ``coolant``, ``pm``) and ``target_variables: [5]`` (``pm``).

The dataset is downloaded from ``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/pmsm_rotor_temp.zip``. It
follows the forecasting layout (``files/`` plus ``annotations/``), see :doc:`/byod/forecasting_format`.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/forecasting_pmsm_rotor_temp/config.yaml

         # MSPM0G5187 variant
         ./run_tinyml_modelzoo.sh examples/forecasting_pmsm_rotor_temp/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\forecasting_pmsm_rotor_temp\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_forecasting
     target_device: F28P55
   dataset:
     dataset_name: forecasting_pmsm_rotor_temp
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/pmsm_rotor_temp.zip
   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     frame_size: 3
     stride_size: 0.4
     forecast_horizon: 1
     variables: 6
     target_variables:
     - 5
   training:
     model_name: FCST_LSTM8
     model_config: ''
     batch_size: 256
     training_epochs: 20
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     quantization: 2
     optimizer: adam
     learning_rate: 0.005
     output_int: false
   testing: {}
   compilation: {}

* ``SimpleWindow`` and ``frame_size`` are mandatory for forecasting; feature extraction is not supported.
* ``output_int`` must be ``false``.
* ``config_MSPM0.yaml`` differs only in the target device (``MSPM0G5187``) and in having no early stopping.

Evaluation
----------

The metrics are SMAPE (0 to 200%, lower is better) and R2 (higher is better). The epoch with the lowest SMAPE
across all targets and predicted time steps is the best epoch. Per-epoch plots and CSV files are written to the
``best_epoch_<n>_results`` and ``test_results`` folders (see below).

Results
-------

Results of the example readme (best epoch statistics, SMAPE and R2 of ``pm``, time step 1):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Stage
     - SMAPE
     - R2
   * - Float train, best epoch 19
     - 0.09%
     - 1.0000
   * - Quantized train, best epoch 10
     - 0.26%
     - 0.9999
   * - Test set
     - 0.25%
     - 0.9998

The automated 1.5.0 config test run of ``config.yaml`` gave a quantized best epoch SMAPE of 0.11% (R2 0.9999) and
a test SMAPE of 0.13% (R2 0.9999), consistent with the readme.

.. figure:: /_static/img/examples/forecasting_pmsm_rotor/pm_predictions.png
   :width: 700px
   :align: center
   :alt: PMSM rotor temperature predictions

   Predicted versus actual permanent magnet temperature (black dotted line is the ideal x = y)

Output Location
---------------

Results are written to ``./data/projects/forecasting_pmsm_rotor_temp/run/<date-time>/FCST_LSTM8/``:

* ``training/base/best_epoch_<n>_results``: float model
* ``training/quantization/best_epoch_<n>_results`` and ``training/quantization/test_results``: quantized model,
  each with ``prediction_plots`` and ``predictions_csv``
* ``compilation/artifacts/``: ``mod.a`` and ``tvmgen_default.h``
* ``training/quantization/golden_vectors/``: ``test_vector.c`` and ``user_input_config.h``

Running on Device
-----------------

The readme compiled the example with the TI-NPU software preset for F28P55 and ran it on a LAUNCHXL-F28P55X
(C2000Ware 6.0.1, CCS 20.2.0). Follow :doc:`/deploying_forecasting_models_from_modelmaker_to_device/readme`. After
flashing, the model output matches the golden vectors.

.. figure:: /_static/img/examples/forecasting_pmsm_rotor/ondevice_results.png
   :width: 500px
   :align: center
   :alt: On-device results

   On-device output matching the golden vectors

Measured on the device (example readme, written for 1.3):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Metric
     - Value
   * - Cycles
     - 102307
   * - Inference time
     - 682.05 us
   * - Code size
     - 4375 bytes
   * - Results match golden vectors
     - TRUE

See Also
--------

* :doc:`hvac_indoor_temp_forecast`
* :doc:`generic_forecasting`
* :doc:`forecasting_example`
