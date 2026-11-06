====================================
Generic Timeseries Anomaly Detection
====================================

A "Hello World" introduction to autoencoder-based timeseries anomaly detection, using a synthetic signal
with four kinds of anomalies.

Example folder: ``examples/generic_timeseries_anomalydetection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/generic_timeseries_anomalydetection>`_)

Overview
--------

* **Task**: timeseries anomaly detection (``generic_timeseries_anomalydetection``), autoencoder trained on normal data only
* **Dataset**: synthetic signal ``y = 1.2 sin(2 pi f t) + 0.8 cos(2 pi f t)``, 100 Hz sampling
* **Configs**:

  * ``config.yaml``: device ``F29H85``, model ``AD_17k``
  * ``config_NPU.yaml``: device ``F28P55`` (NPU), model ``AD_11k_NPU``

The Dataset
-----------

Normal signals have a base frequency of 1 Hz (about +/-5% variation), an amplitude variation of up to +/-10% and
Gaussian noise with a standard deviation of 0.05. The four anomaly types are:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Parameter
     - Normal range
     - Anomaly range
   * - Frequency
     - 0.95 - 1.05 Hz
     - Faster 1.3 - 1.7 Hz, slower 0.5 - 0.8 Hz
   * - Amplitude (sine)
     - 1.08 - 1.32
     - Higher 1.92 - 2.64, lower 0.36 - 0.72
   * - Amplitude (cosine)
     - 0.72 - 0.88
     - Higher 1.28 - 1.76, lower 0.24 - 0.48
   * - Noise std dev
     - 0.03 - 0.07
     - Same for all types

There are 60 normal files and 32 anomaly files (8 per type), 92 in total. Each file has 5000 samples
(50 seconds at 100 Hz). The config downloads
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/generic_timeseries_anomalydetection.zip`` automatically.

Dataset Format and Splitting
----------------------------

.. code-block:: text

   dataset_name/
   |
   |-- classes/
       |-- Normal/
       |     |-- normal_0000.csv ...
       |
       |-- Anomaly/
             |-- freq_faster_0000.csv ...
             |-- amp_lower_0000.csv ...

Each CSV holds a single column of signal values. ModelZoo splits the data automatically with
``dataset.split_factor``:

* **Training**: 50% of the normal files only (the autoencoder learns normal behavior)
* **Validation**: 10% of the normal files only
* **Test**: the remaining 40% of the normal files plus all anomaly files

Why Frame Size Matters
----------------------

Anomalies here are changes of frequency and amplitude, so the model must see at least one full cycle of
the signal. With a 100 Hz signal and 1 Hz base frequency, a frame of about 100 samples covers one cycle.
A single sample cannot distinguish a 1 Hz signal from a 1.5 Hz signal. As a rule of thumb, choose a frame
that spans at least one full cycle of the pattern.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/generic_timeseries_anomalydetection/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\generic_timeseries_anomalydetection\config.yaml

For the NPU variant use ``examples/generic_timeseries_anomalydetection/config_NPU.yaml``.

Configuration
-------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F29H85'
     run_name: '{date-time}/{model_name}'
   dataset:
     enable: True
     dataset_name: generic_timeseries_anomalydetection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/generic_timeseries_anomalydetection.zip
     split_factor: [0.5, 0.1, 0.4]
   data_processing_feature_extraction:
     data_proc_transforms:
     - SimpleWindow
     - Downsample
     frame_size: 100
     sampling_rate: 100
     new_sr: 10
     variables: 1
   training:
     model_name: AD_17k
     model_config: ''
     batch_size: 64
     training_epochs: 50
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     quantization: 1
     output_int: false
   testing:
     enable: True
   compilation:
     enable: True

* ``Downsample`` reduces the data from ``sampling_rate`` to ``new_sr`` before windowing.
* ``output_int`` must be ``false`` for anomaly detection because the reconstruction error is compared to a
  float threshold.
* ``config_NPU.yaml`` differs in the target device (``F28P55``), ``frame_size: 128``,
  ``model_name: AD_11k_NPU``, 30 training epochs (no early stopping), ``quantization: 2`` and
  ``auto_quantization: False``.

How It Works
------------

1. The autoencoder is trained on normal data only and learns to reconstruct it with a low error.
2. After training, ModelZoo computes the mean and standard deviation of the reconstruction error on the
   normal training data and creates thresholds ``mean + k * std`` for a range of ``k`` values.
3. On the test set, a window is flagged as an anomaly if its reconstruction error exceeds the threshold.
   Accuracy, precision, recall, F1 score and false positive rate are reported per ``k``.

Results
-------

The example readme documents a reference run with ``AD_17k`` (float best epoch 46 with validation MSE
0.008; quantized best epoch 5 with validation MSE 0.009):

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Quantity
     - Value (readme run)
   * - Reconstruction error, normal training data (mean / std)
     - 0.007815 / 0.002208
   * - Reconstruction error, normal test data (mean)
     - 0.008051
   * - Reconstruction error, anomaly test data (mean)
     - 0.689792
   * - Threshold at k = 4.5
     - 0.017751
   * - Accuracy / precision / recall / F1 at k = 4.5
     - 99.97% / 99.95% / 100.00% / 99.97%
   * - False positive rate at k = 4.5
     - 0.07% (7 of 9624 normal windows)

The automated 1.5.0 test run of ``config.yaml`` also finished with 100% accuracy and F1 at k = 4.0. The test
run of ``config_NPU.yaml`` failed with a CUDA device mismatch error during quantization-aware training
(a tensor on CPU versus one on ``cuda:0``). Treat the NPU variant as unverified in 1.5.0.

Output Location
---------------

Results are written to ``./data/projects/generic_timeseries_anomalydetection/run/<date-time>/AD_17k/``.
The training folders contain the reconstruction-error plots and the threshold performance table; compiled
artifacts are in ``compilation/artifacts/``. See :doc:`/getting_started/understanding_run_output`.

Running on Device
-----------------

Follow :doc:`/deploying_anomaly_detection_models_from_modelzoo_to_device/readme`.

Next Steps
----------

* :doc:`anomaly_detection_example`
* :doc:`/byod/anomaly_detection_format`
