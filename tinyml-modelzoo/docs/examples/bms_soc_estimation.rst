=============================================
Battery State of Charge (SOC) Estimation
=============================================

Estimate the state of charge of a lithium-ion cell from voltage, current, temperature and accumulated
capacity on the MSPM0G5187 microcontroller with its integrated NPU. New in 1.5.0.

Example folder: ``examples/bms_soc_estimation``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/bms_soc_estimation>`_)

Overview
--------

* **Task**: timeseries regression (``generic_timeseries_regression``)
* **Application**: battery management systems (BMS), prevention of over-charge and over-discharge, remaining
  runtime estimation, battery life extension
* **Device**: MSPM0G5187 (``config_MSPM0.yaml``, the only config in the folder)
* **Model**: ``REGR_20k_NPU`` (about 20,000 parameters, INT8 NPU quantization)
* **Target**: ``SOC``

Traditional coulomb counting and open-circuit-voltage lookup drift over time and are sensitive to sensor noise
and battery aging. A short window of raw voltage, current and temperature is not enough either, because SOC
depends on the charge and discharge history. The model therefore also receives the accumulated charge
(``capacity_Ah``) as an input.

The Dataset
-----------

The dataset is the public LG 18650HG2 Li-ion battery data (``LG_18650HG2_Liion_Battery_Data``), charge and
discharge cycles logged at fixed intervals, split into 4491 files. Each file provides four input variables in
this column order:

#. ``voltage_V``
#. ``current_A``
#. ``temperature_C``
#. ``capacity_Ah``

The target is ``SOC``. The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/LG_18650HG2_Liion_Battery_Data.zip``. The layout is the
regression layout (``files/`` plus ``annotations/``), see :doc:`/byod/regression_format`.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/bms_soc_estimation/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\bms_soc_estimation\config_MSPM0.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: timeseries
     task_type: generic_timeseries_regression
     target_device: MSPM0G5187
     run_name: '{date-time}/{model_name}'
   dataset:
     dataset_name: LG_18650HG2_Liion_Battery_Data
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/LG_18650HG2_Liion_Battery_Data.zip
   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     frame_size: 16
     stride_size: 0.5
     variables: 4
   training:
     enable: true
     model_name: REGR_20k_NPU
     quantization: 2
     num_gpus: 1
     training_epochs: 50
     batch_size: 128
     learning_rate: 0.001
   testing: {}
   compilation: {}

``quantization: 2`` selects INT8 NPU-optimized quantization, and compilation with the TI Neural Network
Compiler generates ``mod.a`` and ``tvmgen_default.h`` for the MSPM0G5187 NPU. ``num_gpus: 1`` expects a GPU;
use 0 for CPU.

Results
-------

The example readme reports the following for ``REGR_20k_NPU`` on the full dataset (best run):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Metric
     - Value
   * - Test RMSE
     - 2.56% SOC (0.0256)
   * - Test R2
     - 0.99
   * - Best-epoch R2 during training
     - 0.987
   * - Flash on MSPM0G5187
     - 26.39 KB (code 2.43 KB, read-only data 21.52 KB, read-write data 2.45 KB)

The readme sets a stretch target of RMSE below 2% SOC with R2 above 0.99.

.. note::

   The readme results were measured with an earlier release. In the automated 1.5.0 config test run
   ``config_MSPM0.yaml`` failed while loading the dataset (the annotation list file was not found, ``IndexError``
   in ``load_list``), so these results are unverified in 1.5.0.

Output Location
---------------

Results are written to ``./data/projects/LG_18650HG2_Liion_Battery_Data/run/<date-time>/REGR_20k_NPU/``. The
device files are ``compilation/artifacts/mod.a``, ``compilation/artifacts/tvmgen_default.h`` and the golden
vectors in ``training/quantization/golden_vectors/``.

Running on Device
-----------------

Use the MSPM0 SDK (2.08.00 or later) with Code Composer Studio and follow
:doc:`/deploying_regression_models_from_modelmaker_to_device/readme` and :doc:`/DEPLOYING_TO_SDK_EXAMPLE`. For
the MSPM0G5187 NPU see :doc:`/devices/index`.

See Also
--------

* :doc:`generic_regression`
* :doc:`washing_machine_regression`
