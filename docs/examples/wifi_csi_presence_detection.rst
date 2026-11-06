============================
Wi-Fi CSI Presence Detection
============================

Detect human presence from Wi-Fi Channel State Information (CSI) on the CC35X1 Wi-Fi microcontroller. This
example is new in 1.5.0.

Example folder: ``examples/wifi_csi_presence_detection``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/wifi_csi_presence_detection>`_)

Overview
--------

The model classifies 2-second windows of CSI data into **no presence** (class 0) and **presence** (class 1),
enabling device-free occupancy sensing without body-worn sensors. PIR sensors need line of sight and
miss stationary occupants, whereas Wi-Fi CSI is pervasive, passive and sensitive to the small environmental
changes caused by a person even at rest. Threshold based CSI detectors are brittle across rooms and furniture
arrangements; a trained model on the Wi-Fi SoC gives always-on, privacy-preserving sensing without extra hardware.

* **Task**: binary classification (``generic_timeseries_classification``)
* **Device**: CC35X1 (`CC3551E <https://www.ti.com/product/CC3551E>`_); the model is compiled for the M33 CDE
* **Model**: ``WiFiPresDet_4.7k_CDE`` (about 4.7K parameters)
* **Data**: CSI magnitudes of 52 usable subcarriers at 128 Hz

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/wifi_csi_presence_detection/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\wifi_csi_presence_detection\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: CC35X1
     run_name: '{date-time}/{model_name}'
   dataset:
     dataset_name: wifi_csi_occupancy_detection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/wifi_presence_detection_dsi.zip
   data_processing_feature_extraction:
     feature_extraction_name: WiFiCSI_128Input_L2Norm2DFFT_64x26Feature_1Frame_52Subcarriers
     stride_size: 0.7
   training:
     model_name: WiFiPresDet_4.7k_CDE
     batch_size: 32
     training_epochs: 50
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     learning_rate: 0.00005
     quantization: 2
     auto_quantization: false
     output_int: false
   testing: {}
   compilation: {}

Datasets
--------

All datasets contain CSI recordings at 128 Hz across 52 usable subcarriers (hardware indices 0 to 25 and 27 to
52; index 26 is the DC subcarrier and is excluded). Labels are binary.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dataset
     - Description
   * - ``wifi_presence_detection_dsi.zip``
     - Compact in-house dataset and the default of ``config.yaml``; already preprocessed. Recorded in an indoor
       conference room below 300 sq ft with a CC35X1 LP receiver and a TP-Link Archer C6 router as transmitter.
       400 windows of 2 s per class (800 total); the presence class covers sitting, standing, walking and waving.
   * - ``wifi_presence_detection_dsd.zip``
     - Larger and more diverse dataset (2 GB+ compressed, over 6.4 million data points, 8 recording days, 13
       activity conditions collapsed to binary). Requires ``preprocess_dsd.py`` before training.
   * - ``wifi_presence_detection_dsd_2000.zip``
     - Uniformly sampled subset of the DSD data with 2000 samples balanced between both classes; already preprocessed.
   * - ``wifi_presence_detection_dsd_4400.zip``
     - Same, with 4400 samples; already preprocessed.

To use the DSD dataset, edit ``IN_ROOT`` and ``OUT_ROOT`` in ``examples/wifi_csi_presence_detection/preprocess_dsd.py``,
run it, and set ``dataset.input_data_path`` to the output directory. ``preprocess_dsd.py`` skips the first 10 s of
each recording, extracts 256-sample (2 s) windows with 30% overlap, resamples them, and assigns the train, validation and
test splits by recording date. ``preprocess.py`` does the equivalent for raw DSI-style CSV folders (drops metadata
columns, resamples each session to 128 Hz).

Feature Extraction
------------------

The preset ``WiFiCSI_128Input_L2Norm2DFFT_64x26Feature_1Frame_52Subcarriers`` converts the 52-subcarrier CSI
magnitude windows into a 2-D time-frequency feature map:

.. list-table::
   :header-rows: 1
   :widths: 8 62 30

   * - Step
     - Operation
     - Output shape
   * - 1
     - Downsample 128 Hz to 64 Hz
     - 64 x 52
   * - 2
     - Sliding window (2 s at 64 Hz, stride 0.7)
     - 128 x 52
   * - 3
     - L2 normalize each time step across subcarriers (``L2_NORM_ROW``)
     - 128 x 52
   * - 4
     - Real FFT along the time axis (``FFT_FE``)
     - 65 x 52
   * - 5
     - Keep the positive half, drop Nyquist (``FFT_POS_HALF``)
     - 64 x 52
   * - 6
     - Magnitude (``ABS``)
     - 64 x 52
   * - 7
     - Log magnitude, log10(1 + abs(F)) (``LOG_DB``)
     - 64 x 52
   * - 8
     - Real FFT across the subcarrier axis, positive half (``FFT_COL``)
     - 64 x 26

The final model input per window has shape ``(1, 26, 1, 64)``.

Results
-------

The example readme reports these results for the DSI dataset (float model 99.17% accuracy, 8-bit weight and
8-bit activation QAT model 99.17%, test set 98.75%, AUC ROC 0.985 on the test set) and, on the test set, 98.85%
(DSD), 99.00% (DSD 2000) and 99.50% (DSD 4400). In the automated 1.5.0 config test run, ``config.yaml`` reached a
test accuracy of 98.75% with an AUC ROC of 0.988.

Training and Deployment
-----------------------

Running the config performs feature extraction, training, 8-bit quantization (``quantization: 2``) and
compilation with the TI Neural Network Compiler for the M33 CDE of the CC35X1, producing ``model.a`` and
``tvmgen_default.h`` for firmware integration. See :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

See Also
--------

* :doc:`pir_detection`: PIR based presence/motion classification
* `CC3551E product page <https://www.ti.com/product/CC3551E>`_
* `TI Neural Network Compiler guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
