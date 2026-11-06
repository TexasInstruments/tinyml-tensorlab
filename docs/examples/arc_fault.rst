========================
DC Arc Fault Detection
========================

DC arc fault detection is one of the main applications of Tiny ML ModelZoo. The example folder
``examples/dc_arc_fault`` contains six configurations: two supervised classification
variants (DSK and DSI datasets), a bring-your-own-model variant, an on-device learning variant, and
two anomaly detection variants.

Example folder: ``examples/dc_arc_fault``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/dc_arc_fault>`_).
For AC arc faults on MSPM0, see :doc:`ac_arc_fault`.

Overview
--------

* **Task**: binary classification (normal vs. arc), ``task_type: arc_fault``; anomaly detection variants use ``generic_timeseries_anomalydetection``
* **Application**: solar inverters, battery systems, DC distribution
* **Datasets**: real current waveforms from arc fault experiments (DSK and DSI captures)
* **Default device**: F28P55 (NPU)

Why Arc Fault Detection?
------------------------

Electrical arcs are a fire hazard in buildings, damage equipment in industrial settings and are a
safety risk in battery systems. Fuses and breakers do not detect low-level arcs reliably, whereas an AI
model can learn the arc signature in the current waveform.

Available Configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 14 46

   * - Config file
     - Device
     - Model
     - Notes
   * - ``config_dsk.yaml``
     - F28P55
     - ``CLS_1k_NPU``
     - DSK dataset ``dc_arc_fault_example_dsk``, preset ``FFT1024Input_256Feature_1Frame_Full_Bandwidth``, 10 epochs, batch size 32
   * - ``config_dsi.yaml``
     - F28P55
     - ``CLS_13k_NPU``
     - DSI dataset ``arc_fault_example_dsi`` (313 kHz), ``Downsample`` + ``SimpleWindow`` transforms, 10 epochs, batch size 2048
   * - ``config_dsk_my_model.yaml``
     - F28P55
     - ``MY_CLS_2k_NPU``
     - Same as ``config_dsk.yaml`` but with a user-defined model; see :doc:`/BRING_YOUR_OWN_MODEL`
   * - ``config_dsi_ondevice_learning.yaml``
     - AM13E2
     - ``CLS_24k_NPU``
     - DSI data with on-device training enabled (``ondevice_training: true``, last 3 layers trainable); see :doc:`/ondevice_training/overview`
   * - ``config_anomaly_detection_dsi.yaml``
     - F28P55
     - ``AD_17k``
     - Anomaly detection on DSI data (``arc_fault_example_dsi_ad``), 100 epochs, ``forced_soft_npu_preset`` compilation
   * - ``config_anomaly_detection_dsk.yaml``
     - F28P55
     - ``AD_17k``
     - Anomaly detection on DSK data (``arc_fault_example_dsk_ad``), 250 epochs

Datasets are downloaded automatically from ``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/``
(for example ``arc_fault_classification_dsi.zip``, ``arc_fault_dsi_anomaly_detection.zip``,
``arc_fault_dsk_anomaly_detection.zip``). The DSK classification dataset is resolved from its dataset name.

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from
the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsk.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\dc_arc_fault\config_dsk.yaml

Replace ``config_dsk.yaml`` with any config from the table above.

Configuration (``config_dsk.yaml``)
-----------------------------------

.. code-block:: yaml

   common:
     task_type: arc_fault
     target_device: F28P55
   dataset:
     dataset_name: dc_arc_fault_example_dsk
   data_processing_feature_extraction:
     feature_extraction_name: FFT1024Input_256Feature_1Frame_Full_Bandwidth
     variables: 1
   training:
     model_name: CLS_1k_NPU
     batch_size: 32
     training_epochs: 10
     num_gpus: 0
     early_stopping: true
     early_stopping_patience: 5
   testing: {}
   compilation: {}

``FFT1024Input_256Feature_1Frame_Full_Bandwidth`` takes a 1024-sample frame, computes the FFT and keeps
256 frequency features of one frame (no temporal concatenation). This captures the high-frequency
signature of arcs.

Configuration for the DSI Dataset
---------------------------------

The DSI captures are sampled at 313 kHz, so the config downsamples them and windows the signal
instead of using a feature extraction preset:

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms: [Downsample, SimpleWindow]
     gain_variations:
       arc: [0.9, 1.1]
       normal: [0.8, 1.2]
     sampling_rate: 313000
     new_sr: 3130
     frame_size: 256
     stride_size: 0.01
     variables: 1
   training:
     model_name: CLS_13k_NPU
     batch_size: 2048
     training_epochs: 10

Available Models
----------------

Any enabled classification model can be used. Commonly used choices:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Approx. parameters
     - Description
   * - ``CLS_500_NPU``
     - 500
     - Small, NPU-optimized
   * - ``CLS_1k_NPU``
     - 1,000
     - Baseline NPU model (default for the DSK config)
   * - ``CLS_2k_NPU``
     - 2,000
     - Medium NPU model
   * - ``CLS_4k_NPU``
     - 4,000
     - Larger NPU model
   * - ``CLS_13k_NPU``
     - 13,000
     - Used for the DSI config
   * - ``ArcFault_model_200_t`` / ``_300_t`` / ``_700_t`` / ``_1400_t``
     - 200 to 1,400
     - Arc fault specific models (also enabled in 1.5.0)

Results from the 1.5.0 Test Run
-------------------------------

The following values come from the automated 1.5.0 config test logs (runs of each config with default settings,
device F28P55 unless noted). They illustrate what to expect; your numbers will vary with the data split and
random seed.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Config
     - Test accuracy
     - AUC ROC
   * - ``config_dsk.yaml`` (``CLS_1k_NPU``)
     - 100.00%
     - 1.000
   * - ``config_dsi.yaml`` (``CLS_13k_NPU``)
     - 94.98%
     - 0.976
   * - ``config_dsi_ondevice_learning.yaml`` (``CLS_24k_NPU``, 1 epoch)
     - 85.13%
     - 0.960
   * - ``config_anomaly_detection_dsi.yaml`` (``AD_17k``)
     - F1 score 100.00%
     - not applicable

.. note::

   In the same test run, ``config_anomaly_detection_dsk.yaml`` failed during quantization-aware training
   (a per-channel fake-quantize error), so no result is reported for it. Treat that config as not yet working.

Interpreting Results
--------------------

After training, check these outputs in the ``run`` folder (see :doc:`/getting_started/understanding_run_output`).
The figures below are examples from the 1.4.0 version of this page and are shown for illustration only.

**ROC curve** (``One_vs_Rest_MultiClass_ROC_test.png``): the AUC should be close to 1.0.

.. figure:: /_static/img/examples/arc_fault/One_vs_Rest_MultiClass_ROC_test.png
   :width: 600px
   :align: center
   :alt: ROC curve for arc fault detection

   One-vs-rest multi-class ROC curves

**Class score histogram** (``Histogram_Class_Score_differences_test.png``): wide separation indicates a
robust classifier.

.. figure:: /_static/img/examples/arc_fault/Histogram_Class_Score_differences_test.png
   :width: 600px
   :align: center
   :alt: Class score histogram

   Distribution of class score differences

**FPR/TPR CSV** (``fpr_tpr_thresholds.csv``): use it to choose the operating threshold, balancing arc
detection against false alarms.

Deployment Considerations
-------------------------

* In safety applications choose the threshold according to whether you prefer sensitivity (catch every
  arc, accept false alarms) or specificity (fewer false alarms, may miss subtle arcs).
* Bring your own model: :doc:`/BRING_YOUR_OWN_MODEL`.
* Deploy the compiled artifacts into an SDK example: :doc:`/DEPLOYING_TO_SDK_EXAMPLE`
  (see also :doc:`/deployment/npu_device_deployment`).

Next Steps
----------

* Try your own data: :doc:`/byod/classification_format`
* AC arc faults on MSPM0G5187: :doc:`ac_arc_fault`
* Another example: :doc:`motor_bearing_fault`
