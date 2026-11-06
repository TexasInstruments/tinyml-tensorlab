===================================
HVAC Indoor Temperature Forecasting
===================================

Forecast the indoor temperature of a room from compressor frequency and outdoor temperature, as an input to
predictive HVAC control.

Example folder: ``examples/hvac_indoor_temp_forecast``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/hvac_indoor_temp_forecast>`_)

Overview
--------

Traditional HVAC controllers react to temperature deviations after they occur, which causes delayed
corrections, unnecessary compressor activity and higher energy use. A forecast of the indoor temperature lets a
higher-level controller plan smoother, more energy-efficient compressor actions. This example trains a small
LSTM offline and compiles it to run fully on the device.

* **Task**: timeseries forecasting (``generic_timeseries_forecasting``)
* **Device**: F28P55 (``config.yaml``), compiled with the ``forced_soft_npu_preset`` (software emulation of the NPU)
* **Model**: ``FCST_LSTM10``
* **Target**: ``indoorTemperature`` at the next time step (set ``forecast_horizon`` for a longer horizon)

The Dataset
-----------

A synthetic dataset that emulates indoor thermal behavior. It has uniformly sampled values of:

#. ``compressorFrequency``
#. ``outdoorTemperature``
#. ``indoorTemperature``

The model input is the past 5 values of each of the three signals (``frame_size: 5``, ``variables: 3``) and the
target is ``indoorTemperature`` (column index 2, counting from 0 after the time column). The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/hvac_indoor_temp_forecast_dataset.zip``. The layout is
the forecasting layout described in :doc:`/byod/forecasting_format`.

.. note::

   In the automated 1.5.0 config test run this config failed while extracting the download
   ("File is not a zip file"). If you see the same error, check that the dataset URL returns a valid zip
   before debugging the pipeline.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/hvac_indoor_temp_forecast/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\hvac_indoor_temp_forecast\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_forecasting
     target_device: F28P55
   dataset:
     dataset_name: hvac_indoor_temp_forecast
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/hvac_indoor_temp_forecast_dataset.zip
   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     frame_size: 5
     stride_size: 0.2
     forecast_horizon: 1
     variables: 3
     target_variables:
     - 2
   training:
     model_name: FCST_LSTM10
     model_config: ''
     batch_size: 6
     training_epochs: 10
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     quantization: 1
     optimizer: adamw
     learning_rate: 0.003
     weight_decay: 0.001
     lr_scheduler: none
     output_int: false
   testing: {}
   compilation:
     compile_preset_name: forced_soft_npu_preset

``target_variables`` also accepts column names or several indices. ``output_int`` must be ``false`` for
forecasting.

Results
-------

The example readme reports these results for the indoor temperature (time step 1):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Stage
     - SMAPE
     - R2
   * - Float train, best epoch
     - 0.30%
     - 0.9971
   * - Quantized train, best epoch 4
     - 0.80%
     - 0.9893
   * - Test set
     - 0.73%
     - 0.9857

.. warning::

   These numbers come from the example readme and were not reproduced in 1.5.0 because the test run could not
   download the dataset. Quantized ``FCST_LSTM10`` models have shown a large accuracy drop in other 1.5.0 runs
   (see :doc:`generic_forecasting`), so compare the float and quantized results of your own run.

.. figure:: /_static/img/examples/hvac_indoor_temp/indoorTemperature_predictions.png
   :width: 700px
   :align: center
   :alt: Predicted versus actual indoor temperature

   Predicted versus actual indoor temperature

Output Location
---------------

Results are written to ``./data/projects/hvac_indoor_temp_forecast/run/<date-time>/FCST_LSTM10/``. The device
files are ``compilation/artifacts/mod.a``, ``compilation/artifacts/tvmgen_default.h`` and the golden vectors in
``training/quantization/golden_vectors/``.

Running on Device
-----------------

Follow :doc:`/deploying_forecasting_models_from_modelmaker_to_device/readme`. After flashing, the model output
matches the golden vectors.

.. figure:: /_static/img/examples/hvac_indoor_temp/ondevice_results.png
   :width: 500px
   :align: center
   :alt: On-device results

   On-device output matching the golden vectors

Device measurements from the readme (F28P55x, written for an earlier release):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Metric
     - Value
   * - AI model cycles
     - 199432
   * - Inference time
     - 1329.55 us
   * - Model size (flash / SRAM)
     - 9174 bytes / 278 bytes
   * - Results match golden vectors
     - TRUE

See Also
--------

* :doc:`forecasting_pmsm_rotor`
* :doc:`generic_forecasting`
