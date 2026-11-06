==============================
Generic Timeseries Forecasting
==============================

A "Hello World" introduction to timeseries forecasting with Tiny ML ModelZoo, using a simulated thermostat
dataset.

Example folder: ``examples/generic_timeseries_forecasting``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/generic_timeseries_forecasting>`_)

Overview
--------

* **Task**: timeseries forecasting (``generic_timeseries_forecasting``), predict future values from past values
* **Dataset**: simulated thermostat, room temperature controlled by an ON/OFF heater with hysteresis
* **Device**: F28P55 (``config.yaml``)
* **Model**: ``FCST_LSTM10`` (single-layer LSTM with hidden size 10, 542 trainable parameters according to the readme)

The Dataset
-----------

The heater turns ON when the temperature drops below 20 C and OFF when it rises above 24 C. Because of
thermal inertia the temperature oscillates gradually, which makes it a good first forecasting problem.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``timestamp``
     - Time index
   * - ``temperature``
     - Room temperature in degrees C

The split is 10 training files (``thermostat_01.csv`` to ``thermostat_10.csv``), 2 validation files
(``thermostat_11.csv``, ``thermostat_12.csv``) and 3 test files (``thermostat_13.csv`` to
``thermostat_15.csv``). Each file has 1000 time steps. The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/generic_timeseries_forecasting.zip`` automatically.

Dataset Format
--------------

Forecasting always requires annotation files that assign each file to the train, validation or test set.
The data folder is named ``files``.

.. code-block:: text

   {dataset_name}.zip/
   |
   |-- files/
   |     |-- {file1}.csv
   |     |-- {fileN}.csv
   |
   |-- annotations/
         |-- file_list.txt
         |-- instances_train_list.txt
         |-- instances_val_list.txt
         |-- instances_test_list.txt

To use your own data, see :doc:`/byod/forecasting_format`.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/generic_timeseries_forecasting/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\generic_timeseries_forecasting\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_forecasting
     target_device: F28P55
   dataset:
     dataset_name: generic_timeseries_forecasting
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/generic_timeseries_forecasting.zip
   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow        # mandatory for forecasting
     frame_size: 32
     stride_size: 0.1
     forecast_horizon: 2   # number of future time steps to predict
     variables: 1          # only the temperature column (after the timestamp)
     target_variables:
     - 0                   # column index 0 = temperature
   training:
     model_name: FCST_LSTM10
     model_config: ''
     batch_size: 32
     training_epochs: 50
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     quantization: 2
     optimizer: adam
     output_int: false
   testing: {}
   compilation: {}

Key points for forecasting configs:

* ``SimpleWindow`` must be in ``data_proc_transforms`` and ``frame_size`` is mandatory.
  ``stride_size`` is a fraction of the frame size and ``forecast_horizon`` is the number of future steps.
* ``variables`` is the number of input columns after the time column. ``target_variables`` selects what to
  predict. It accepts a list of column indices (0-based after the time column), a list of column names, a
  single index or name, a comma-separated string of names, or an empty list to predict all columns.
* ``output_int`` must be ``false`` for forecasting.
* Feature extraction is not supported for forecasting.

Evaluation Metrics
------------------

**SMAPE (symmetric mean absolute percentage error)**: percentage error normalized to avoid bias toward large
or small values. Range 0% to 200%, lower is better.

**R2 score**: how well predictions match actual values. Range (-infinity, 1], higher is better.

For each epoch SMAPE is computed across all target variables and all predicted time steps; the epoch with the
lowest average SMAPE is the best epoch. R2 and SMAPE are reported per time step of each target variable.

Results
-------

The example readme reports these float-model results (best epoch 48 in that run):

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Metric
     - Timestep 1
     - Timestep 2
     - Overall
   * - SMAPE
     - 0.60%
     - 0.88%
     - 0.74%
   * - R2
     - 0.9811
     - 0.9563
     - 0.9687

.. warning::

   Quantized accuracy is not reliable in the current release. The readme lists quantized results close to the
   float ones, but in the automated 1.5.0 config test run (short, early stopped) the quantized
   ``FCST_LSTM10`` model reached only an R2 of 0.74 and a SMAPE of 2.23% on the validation set and a
   test SMAPE of about 12.9%, far worse than the float model. Always compare the float and quantized results
   in your own run before deploying.

Output Location
---------------

Results are written to ``./data/projects/generic_timeseries_forecasting/run/<date-time>/FCST_LSTM10/``.

* ``training/base/best_epoch_<n>_results``: float model best epoch results
* ``training/quantization/best_epoch_<n>_results`` and ``training/quantization/test_results``: quantized model
* ``compilation/artifacts/``: ``mod.a`` and ``tvmgen_default.h``

See :doc:`/getting_started/understanding_run_output` for details.

Running on Device
-----------------

Follow :doc:`/deploying_forecasting_models_from_modelmaker_to_device/readme`.

Next Steps
----------

* :doc:`forecasting_pmsm_rotor`: real-world motor temperature forecasting
* :doc:`hvac_indoor_temp_forecast`: indoor temperature forecasting
* :doc:`/byod/forecasting_format`: use your own forecasting data
