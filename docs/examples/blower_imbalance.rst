================
Blower Imbalance
================

Detect blade imbalance in HVAC blowers using 3-variable (3-phase motor current) time series data.

Example folder: ``examples/blower_imbalance``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/blower_imbalance>`_)

Overview
--------

Blade imbalance in HVAC blowers causes increased vibration, noise and premature bearing wear. This
example uses signal analysis on a 3-channel input to detect imbalance conditions before they cause
equipment failure.

* **Application**: HVAC systems, industrial fans, predictive maintenance
* **Task**: time series classification (``task_type: motor_fault``)
* **Data**: multivariate, 3 input variables (``variables: 3``)
* **Dataset**: ``fan_blower_imbalance_dsh`` (resolved from the dataset name)
* **Device**: F28P55 (``config.yaml``)
* **Model**: ``CLS_1k_NPU``

The example folder contains a single configuration. Its ``readme.md`` describes that configuration but not the
dataset class labels or sensor details; inspect the downloaded dataset under
``data/projects/blower_imbalance/dataset`` after the first run (the class names are the folder names
under ``classes/``).

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/blower_imbalance/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\blower_imbalance\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: motor_fault
     target_device: F28P55
   dataset:
     dataset_name: fan_blower_imbalance_dsh
   data_processing_feature_extraction:
     data_proc_transforms: []
     feature_extraction_name: Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1
     variables: 3
   training:
     model_name: CLS_1k_NPU
     batch_size: 256
     training_epochs: 20
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
   testing: {}
   compilation: {}

The preset ``Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1`` uses 256-sample frames per
channel, FFT binned to 16 features, 8 concatenated frames, with DC removed.

Models
------

The ``FanImbalance_model_1_t``, ``FanImbalance_model_2_t`` and ``FanImbalance_model_3_t`` models are enabled in
1.5.0 and can be used instead of the generic NPU models (``CLS_1k_NPU``, ``CLS_2k_NPU``, ``CLS_4k_NPU``):

.. code-block:: yaml

   training:
     model_name: FanImbalance_model_1_t

Results from the 1.5.0 Test Run
-------------------------------

In the automated 1.5.0 config test run, ``config.yaml`` (``CLS_1k_NPU``, 960 parameters) reached a test
accuracy of 99.46% with an AUC ROC of 1.000. Your numbers will vary with the data split and random seed.

See Also
--------

* :doc:`fan_blade_fault_classification`: accelerometer-based fan fault detection
* :doc:`motor_bearing_fault`: the same feature extraction preset on bearing vibration data
