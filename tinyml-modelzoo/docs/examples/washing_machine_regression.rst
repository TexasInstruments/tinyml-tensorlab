=======================================
Washing Machine Load Weighing
=======================================

Estimate the weight of the clothes in a washing machine from motor voltage, current and speed, which removes
the need for a mechanical weight sensor.

In 1.4.0 the example folder was named ``reg_washing_machine``. In 1.5.0 it is ``washing_machine_load_weighing``.
The docname of this page stays ``washing_machine_regression``.

Example folder: ``examples/washing_machine_load_weighing``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/washing_machine_load_weighing>`_)

Overview
--------

* **Task**: timeseries regression (``generic_timeseries_regression``)
* **Application**: sensorless load weighing for automatic water level adjustment, lowering manufacturing cost
* **Configs and devices**:

  * ``config.yaml``: F28P55, ``frame_size: 512``
  * ``config_MSPM0.yaml``: MSPM0G5187, ``frame_size: 1024``

* **Model**: ``REGR_13k`` (12,477 trainable parameters according to the readme); ``REGR_4k`` (4,093 parameters) can be used as a smaller alternative

The Dataset
-----------

The data comes from TI Kilby Labs: 6 voltage, current and speed based signals from a washing machine running
in constant torque mode, with loads from 0 g to 900 g in 100 g steps.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``Vd``
     - Voltage component along the d-axis
   * - ``Id``
     - Current component along the d-axis
   * - ``Vq``
     - Voltage component along the q-axis
   * - ``Iq``
     - Current component along the q-axis
   * - ``Iqref``
     - Reference current
   * - ``Speed``
     - Speed of the washing machine motor

The dataset has 391 files of about 12,300 data points each: 276 for training, 51 for validation and 64 for
testing (70% / 14% / 16%). The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/washing_machine_loading_data.zip``. It follows the
regression layout (``files/`` plus ``annotations/``) described in :doc:`/byod/regression_format`.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/washing_machine_load_weighing/config.yaml

         # MSPM0G5187 variant
         ./run_tinyml_modelzoo.sh examples/washing_machine_load_weighing/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\washing_machine_load_weighing\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_regression
     target_device: F28P55
   dataset:
     dataset_name: washing_machine_load_weighing
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/washing_machine_loading_data.zip
   data_processing_feature_extraction:
     stride_size: 0.1
     data_proc_transforms:
     - SimpleWindow
     frame_size: 512
     variables: 6
   training:
     optimizer: adam
     lr_scheduler: cosineannealinglr
     model_name: REGR_13k
     batch_size: 128
     training_epochs: 100      # use 500 epochs for a better R2 score and RMSE
     early_stopping: true
     early_stopping_patience: 5
     lambda_reg: 0.01
     num_gpus: 1
     quantization: 2
     auto_quantization: True
   testing: {}
   compilation: {}

* Feature extraction such as FFT, binning or ABS did not improve the RMSE for this task, so only ``SimpleWindow``
  with a suitable frame size is used. ``variables: 6`` is the number of input signals.
* ``num_gpus: 1`` expects a GPU; use 0 for CPU.
* ``config_MSPM0.yaml`` targets ``MSPM0G5187``, uses ``frame_size: 1024`` and has no early stopping.
* The training loss is MSE and the reported metric is RMSE. The readme recommends about 500 epochs for a good
  RMSE.
* With ``quantization: 2`` and ``auto_quantization: True`` the model is partially quantized. The input
  BatchNorm, the first conv or linear layer and the last linear layer stay in float, and all other layers
  are INT8. ``quantization: 0`` gives the float model.

Results
-------

Test-set RMSE in grams from the example readme, after 500 training epochs each:

.. list-table::
   :header-rows: 1
   :widths: 16 21 21 21 21

   * - Frame size
     - ``REGR_4k``
     - ``REGR_13k``
     - Partially quantized ``REGR_4k``
     - Partially quantized ``REGR_13k``
   * - 256
     - 63.00
     - 35.77
     - 62.32
     - 33.37
   * - 512
     - 46.91
     - 32.58
     - 52.42
     - 31.23
   * - 1024
     - 59.01
     - 25.78
     - 64.59
     - 36.78

For the 13k model the RMSE falls as the frame size grows, because the model sees more context. The 4k model has
too few parameters to exploit larger frames. The readme quotes a best RMSE of 25.78 g for the float model and
31.23 g for the partially quantized model. These numbers are from the example readme; they were not
regenerated for 1.5.0. The automated 1.5.0 config test run (short, early stopped) completed for both
``config.yaml`` and ``config_MSPM0.yaml``.

.. figure:: /_static/img/examples/washing_machine/float_actual_vs_predicted.png
   :width: 60%
   :align: center
   :alt: Float REGR_13k predicted versus actual weight

   Float ``REGR_13k``, frame size 512 (from the example readme)

.. figure:: /_static/img/examples/washing_machine/partially_quantized_actual_vs_predicted.png
   :width: 60%
   :align: center
   :alt: Partially quantized REGR_13k predicted versus actual weight

   Partially quantized ``REGR_13k``, frame size 512 (from the example readme)

Output Location
---------------

Results are written to ``./data/projects/washing_machine_load_weighing/run/<date-time>/REGR_13k/``. The four
device files are ``compilation/artifacts/mod.a``, ``compilation/artifacts/tvmgen_default.h``,
``training/quantization/golden_vectors/test_vector.c`` and
``training/quantization/golden_vectors/user_input_config.h``.

Running on Device
-----------------

See :doc:`/deploying_regression_models_from_modelmaker_to_device/readme` and :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

See Also
--------

* :doc:`generic_regression`
* :doc:`torque_measurement_regression`
