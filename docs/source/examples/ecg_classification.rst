==================
ECG Classification
==================

Classify normal vs anomalous heartbeats from ECG signals.

Overview
--------

This example demonstrates heartbeat classification using ECG (electrocardiogram)
signals. It can identify normal heartbeats and various arrhythmia conditions,
enabling early detection of cardiac abnormalities.

**Application**: Wearable health monitors, cardiac monitoring, medical devices

**Task Type**: Time Series Classification / Anomaly Detection

**Data Type**: Multivariate (ECG leads)

Configuration
-------------

**Classification Mode:**

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'ecg_classification'

   training:
     model_name: 'CLS_4k_NPU'
     training_epochs: 50
     batch_size: 32

   testing: {}
   compilation: {}

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         # Classification mode
         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/ecg_classification/config.yaml

         # Anomaly detection mode
         ./run_tinyml_modelzoo.sh examples/ecg_classification/config_anomaly_detection.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         # Classification mode
         run_tinyml_modelzoo.bat examples\ecg_classification\config.yaml

         # Anomaly detection mode
         run_tinyml_modelzoo.bat examples\ecg_classification\config_anomaly_detection.yaml

Dataset Details
---------------

**Input Variables**:

* ECG signal samples
* Multiple leads (if available)

**Classes** (Classification mode):

* Normal heartbeat
* Abnormal heartbeat / Arrhythmia

**Anomaly Detection Mode**:

Train on normal heartbeats only, detect anomalies based on reconstruction error.

See Also
--------

* :doc:`anomaly_detection_example` - Anomaly detection tutorial
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ecg_classification>`_
