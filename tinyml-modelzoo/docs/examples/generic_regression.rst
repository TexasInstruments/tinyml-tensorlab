==============================
Generic Timeseries Regression
==============================

A "Hello World" introduction to timeseries regression with Tiny ML ModelZoo. It shows how any generic
timeseries regression task is set up, trained, quantized and compiled.

Example folder: ``examples/generic_timeseries_regression``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/generic_timeseries_regression>`_)

Overview
--------

* **Task**: timeseries regression (``generic_timeseries_regression``), continuous value prediction
* **Dataset**: synthetic, ``y = 1.2 sin(x) + 3.2 cos(x)`` with ``x`` drawn randomly from [0, 3]
* **Device**: F28P55 (``config.yaml``)
* **Model**: ``REGR_2k`` (a small CNN regressor)

The Dataset
-----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Description
   * - ``x``
     - Randomly generated in the range [0, 3]
   * - ``y``
     - Target, ``y = 1.2 sin(x) + 3.2 cos(x)``

The split is 7 training files, 2 validation files and 1 test file, with 5000 data points per file. The
config downloads the dataset automatically from
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/generic_timeseries_regression.zip``.

Dataset Format
--------------

For regression ModelMaker always requires annotation files, which assign each file to the train, validation
or test set. The data folder is named ``files`` for regression and forecasting tasks.

.. code-block:: text

   {dataset_name}.zip/
   |
   |-- files/
   |     |-- {file1}
   |     |-- {file2}
   |     |-- {fileN}
   |
   |-- annotations/
         |-- file_list.txt
         |-- instances_train_list.txt
         |-- instances_val_list.txt
         |-- instances_test_list.txt

To use your own data, see :doc:`/byod/regression_format`.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/generic_timeseries_regression/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\generic_timeseries_regression\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F28P55
   dataset:
     dataset_name: generic_timeseries_regression
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/generic_timeseries_regression.zip
   data_processing_feature_extraction:
     stride_size: 1
     data_proc_transforms:
     - SimpleWindow
     frame_size: 10
     variables: 1
   training:
     optimizer: adam
     lr_scheduler: cosineannealinglr
     model_name: REGR_2k
     batch_size: 256
     training_epochs: 45
     early_stopping: true
     early_stopping_patience: 5
     lambda_reg: 0.01
     num_gpus: 0
     quantization: 2          # 0 = float model, 2 = 8-bit quantized model
     auto_quantization: True
   testing: {}
   compilation: {}

``variables`` is the number of input columns (after the time columns) that are used to predict the target.
``num_gpus: 0`` runs on CPU; set it to 1 if a GPU is available. The ``quantization`` and
``auto_quantization`` keys select the float, partially quantized or fully quantized flow (see below). For
the meaning of every key see :doc:`/CONFIG_YAML_GUIDE`.

Evaluation Metrics
------------------

**RMSE (root mean square error)**: the square root of the mean squared error. Range 0 to infinity, ideal
value 0 (lower is better).

**R2 score (coefficient of determination)**: the fraction of the variance in the data that the model
explains. Range (-infinity, 1], ideal value 1 (higher is better).

The best epoch is the one with the lowest validation MSE; the checkpoint is updated whenever a lower value
is seen.

Results
-------

The example readme reports these test-set results for the three quantization settings:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Model
     - RMSE
     - R2 score
   * - Float (``quantization: 0``, ``auto_quantization: False``)
     - 0.15
     - 0.95
   * - Partially quantized (``quantization: 2``, ``auto_quantization: True``)
     - 0.11
     - 0.98
   * - Fully quantized (``quantization: 2``, ``auto_quantization: False``)
     - 0.11
     - 0.97

These numbers come from the example readme. The automated 1.5.0 config test run (short, early stopped)
reached a quantized test RMSE of 0.33 and R2 of 0.78, so expect results to vary with the training length
and early stopping.

Output Location
---------------

Results are written to ``./data/projects/generic_timeseries_regression/run/<date-time>/REGR_2k/``.

* ``training/quantization/post_training_analysis/``: predicted-versus-actual plot, error histogram and
  ``results_on_test_set.csv`` (columns ``ground_truth`` and ``predicted``)
* ``training/quantization/golden_vectors/``: ``user_input_config.h`` and ``test_vector.c``
* ``compilation/artifacts/``: ``mod.a`` and ``tvmgen_default.h``

See :doc:`/getting_started/understanding_run_output` for details.

Running on Device
-----------------

Follow :doc:`/deploying_regression_models_from_modelmaker_to_device/readme` to integrate the compiled
artifacts into a device project.

Next Steps
----------

* :doc:`torque_measurement_regression`: real-world motor torque prediction
* :doc:`/byod/regression_format`: use your own regression data
* :doc:`/getting_started/understanding_config`: all config options
