==================
ECG Classification
==================

Edge AI solution for real-time ECG signal classification on the MSPM0G5187 with NPU.
Classifies ECG signals into cardiac conditions for portable, low-power heart monitoring.

Overview
--------

Cardiovascular diseases (CVDs) are the leading cause of death globally, responsible
for approximately 17.9 million deaths per year. Early detection through continuous
ECG monitoring significantly improves patient outcomes. However, traditional ECG
analysis requires expensive medical equipment and trained professionals, limiting
access to timely diagnosis.

This example demonstrates how Edge AI enables real-time cardiac classification on
portable, low-power devices -- without the need for cloud connectivity. By deploying
a CNN-based model directly on the MSPM0G5187 microcontroller with NPU acceleration,
ECG signals can be classified into multiple cardiac conditions at the point of care.

* **Application**: Wearable health monitors, portable cardiac monitoring devices
* **Task Type**: Time Series Classification
* **Primary Device**: MSPM0G5187 (with NPU)
* **Also Supported**: AM13E2, F28P55

.. note::

   Device-specific configuration files are provided as ``config_<device>.yaml``.
   See the :ref:`ecg-available-configs` section below for details.

Key Performance Targets
-----------------------

* Real-time ECG classification on-device
* Low power consumption suitable for wearables
* Greater than 95% classification accuracy
* LED feedback for classification results

System Components
-----------------

**Hardware**

* `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ -- Microcontroller with integrated NPU
* `AFE1594 Analog Front End <https://www.ti.com/drr/opn/AFE159RP4-DESIGN>`_ -- ECG signal acquisition

**Software**

* Code Composer Studio (CCS) 12.x or later
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_ 2.08.00 or later
* TI Edge AI Studio

Dataset Details
---------------

**Dataset**: ``ecg_classification_4class``

The dataset contains ECG recordings labeled into the following classes:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - **Normal**
     - Healthy sinus rhythm
   * - **Mild**
     - Minor cardiac abnormalities
   * - **Other**
     - Other cardiac conditions

Feature Extraction
------------------

The feature extraction pipeline for MSPM0 (preset ``ECG2500Input_Roundoff_1Frame``)
processes the raw ECG signal as follows:

1. **Sampling**: 2500 samples per frame from the ECG input
2. **Normalization**: Signal normalization by rounding off
3. **Framing**: Single frame input to the model

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: ECG2500Input_Roundoff_1Frame

Model
-----

**ECG_55k_NPU (Default)**

A CNN architecture optimized for the MSPM0G5187 NPU, with approximately 55K
parameters. This model is designed for NPU-compatible inference, enabling
efficient real-time classification on the target device.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Model
     - Parameters
     - Description
   * - ``ECG_55k_NPU``
     - ~55,000
     - CNN optimized for NPU, default for MSPM0

**Expected Accuracy**: ~97%

Training Configuration
----------------------

The default training hyperparameters for the MSPM0 configuration:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Parameter
     - Value
   * - Batch size
     - 12
   * - Learning rate
     - 0.001
   * - Optimizer
     - Adam
   * - Weight decay
     - 4e-5
   * - Epochs
     - 25

**Quantization**: INT8 quantization is enabled by default for deployment.

**Compilation**: Uses the `TI Neural Network Compiler (NNC) <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_,
optimized for the MSPM0G5187 NPU.

Configuration (MSPM0)
---------------------

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'ecg_classification'
     target_device: 'MSPM0G5187'

   dataset:
     dataset_name: ecg_classification_4class

   data_processing_feature_extraction:
     feature_extraction_name: ECG2500Input_Roundoff_1Frame

   training:
     model_name: ECG_55k_NPU
     batch_size: 12
     learning_rate: 0.001
     weight_decay: 4e-5
     optimizer: 'adam'
     training_epochs: 25
     num_gpus: 0

   testing:
     enable: True

   compilation:
     enable: True

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         # MSPM0G5187 (primary target)
         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/ecg_classification/config_MSPM0.yaml

         # AM13E2
         ./run_tinyml_modelzoo.sh examples/ecg_classification/config.yaml

         # Anomaly detection mode
         ./run_tinyml_modelzoo.sh examples/ecg_classification/config_anomaly_detection.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo

         REM MSPM0G5187 (primary target)
         .\run_tinyml_modelzoo.bat examples\ecg_classification\config_MSPM0.yaml

         REM AM13E2
         .\run_tinyml_modelzoo.bat examples\ecg_classification\config.yaml

         REM Anomaly detection mode
         .\run_tinyml_modelzoo.bat examples\ecg_classification\config_anomaly_detection.yaml

.. _ecg-available-configs:

Available Configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Config File
     - Target Device
     - Mode
   * - ``config_MSPM0.yaml``
     - MSPM0G5187
     - Classification (4-class)
   * - ``config.yaml``
     - AM13E2
     - Classification
   * - ``config_anomaly_detection.yaml``
     - F28P55
     - Anomaly Detection

Anomaly Detection Mode
----------------------

A separate anomaly detection configuration is available for ECG monitoring using
the ``config_anomaly_detection.yaml`` file. This mode trains on normal ECG
recordings only and detects anomalies based on reconstruction error, which can be
useful when labeled fault data is not available.

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'ecg_ad'

   training:
     model_name: 'AD_Linear'
     training_epochs: 200

References
----------

* `MSPM0G5187 Product Page <https://www.ti.com/product/MSPM0G5187>`_
* `AFE1594 Design Resources <https://www.ti.com/drr/opn/AFE159RP4-DESIGN>`_
* `TI Neural Network Compiler (NNC) User's Guide <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_
* `tinyml-tensorlab on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main>`_

See Also
--------

* :doc:`anomaly_detection_example` -- Anomaly detection tutorial
* `Example on GitHub <https://github.com/TexasInstruments/tinyml-tensorlab/tree/main/tinyml-modelzoo/examples/ecg_classification>`_
