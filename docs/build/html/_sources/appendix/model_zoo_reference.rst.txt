===================
Model Zoo Reference
===================

Complete reference of all available models in Tiny ML Tensorlab's model zoo.

Classification Models
---------------------

Standard Classification
^^^^^^^^^^^^^^^^^^^^^^^

For non-NPU devices:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model Name
     - Parameters
     - Description
   * - ``CLS_100``
     - ~100
     - Minimal, very simple tasks
   * - ``CLS_500``
     - ~500
     - Small, simple classification
   * - ``CLS_1k``
     - ~1,000
     - Baseline classification
   * - ``CLS_2k``
     - ~2,000
     - Medium complexity
   * - ``CLS_4k``
     - ~4,000
     - Good accuracy
   * - ``CLS_6k``
     - ~6,000
     - Higher accuracy
   * - ``CLS_13k``
     - ~13,000
     - Complex classification

NPU Classification
^^^^^^^^^^^^^^^^^^

For NPU-enabled devices (F28P55, AM13E2, MSPM0G5187):

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model Name
     - Parameters
     - Description
   * - ``CLS_100_NPU``
     - ~100
     - Minimal NPU model
   * - ``CLS_500_NPU``
     - ~500
     - Small NPU model
   * - ``CLS_1k_NPU``
     - ~1,000
     - Baseline NPU model
   * - ``CLS_2k_NPU``
     - ~2,000
     - Medium NPU model
   * - ``CLS_4k_NPU``
     - ~4,000
     - Recommended NPU model
   * - ``CLS_6k_NPU``
     - ~6,000
     - Larger NPU model
   * - ``CLS_13k_NPU``
     - ~13,000
     - Large NPU model
   * - ``CLS_20k_NPU``
     - ~20,000
     - Very large NPU model
   * - ``CLS_55k_NPU``
     - ~55,000
     - Maximum NPU model

Application-Specific Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Arc Fault Models:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Model Name
     - Parameters
     - Description
   * - ``ArcFault_model_200_t``
     - ~200
     - Minimal arc fault detection
   * - ``ArcFault_model_400_t``
     - ~400
     - Balanced arc fault
   * - ``ArcFault_model_800_t``
     - ~800
     - Higher accuracy arc fault
   * - ``ArcFault_model_1400_t``
     - ~1,400
     - Maximum arc fault accuracy

**Motor Fault Models:**

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Model Name
     - Parameters
     - Description
   * - ``MotorFault_model_1_t``
     - ~1,000
     - Baseline motor fault
   * - ``MotorFault_model_2_t``
     - ~2,000
     - Improved motor fault
   * - ``MotorFault_model_3_t``
     - ~4,000
     - Best motor fault accuracy

Regression Models
-----------------

Standard Regression
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model Name
     - Parameters
     - Description
   * - ``REGR_500``
     - ~500
     - Small regression
   * - ``REGR_1k``
     - ~1,000
     - Baseline regression
   * - ``REGR_2k``
     - ~2,000
     - Medium regression
   * - ``REGR_4k``
     - ~4,000
     - Good accuracy
   * - ``REGR_8k``
     - ~8,000
     - Higher accuracy

NPU Regression
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model Name
     - Parameters
     - Description
   * - ``REGR_500_NPU``
     - ~500
     - Small NPU regression
   * - ``REGR_1k_NPU``
     - ~1,000
     - Baseline NPU regression
   * - ``REGR_2k_NPU``
     - ~2,000
     - Medium NPU regression
   * - ``REGR_4k_NPU``
     - ~4,000
     - Recommended NPU regression
   * - ``REGR_8k_NPU``
     - ~8,000
     - Large NPU regression
   * - ``REGR_20k_NPU``
     - ~20,000
     - Maximum NPU regression

Anomaly Detection Models
------------------------

Autoencoder-based models for anomaly detection.

Standard AD
^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model Name
     - Parameters
     - Description
   * - ``AD_500``
     - ~500
     - Small autoencoder
   * - ``AD_1k``
     - ~1,000
     - Baseline autoencoder
   * - ``AD_2k``
     - ~2,000
     - Medium autoencoder
   * - ``AD_4k``
     - ~4,000
     - Good complexity
   * - ``AD_8k``
     - ~8,000
     - Higher complexity

NPU AD
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model Name
     - Parameters
     - Description
   * - ``AD_500_NPU``
     - ~500
     - Small NPU autoencoder
   * - ``AD_1k_NPU``
     - ~1,000
     - Baseline NPU autoencoder
   * - ``AD_2k_NPU``
     - ~2,000
     - Medium NPU autoencoder
   * - ``AD_4k_NPU``
     - ~4,000
     - Recommended NPU autoencoder
   * - ``AD_8k_NPU``
     - ~8,000
     - Large NPU autoencoder
   * - ``AD_20k_NPU``
     - ~20,000
     - Maximum NPU autoencoder

Forecasting Models
------------------

Standard Forecasting
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model Name
     - Parameters
     - Description
   * - ``FCST_500``
     - ~500
     - Small forecasting
   * - ``FCST_1k``
     - ~1,000
     - Baseline forecasting
   * - ``FCST_2k``
     - ~2,000
     - Medium forecasting
   * - ``FCST_4k``
     - ~4,000
     - Good accuracy
   * - ``FCST_8k``
     - ~8,000
     - Higher accuracy

NPU Forecasting
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Model Name
     - Parameters
     - Description
   * - ``FCST_500_NPU``
     - ~500
     - Small NPU forecasting
   * - ``FCST_1k_NPU``
     - ~1,000
     - Baseline NPU forecasting
   * - ``FCST_2k_NPU``
     - ~2,000
     - Medium NPU forecasting
   * - ``FCST_4k_NPU``
     - ~4,000
     - Recommended NPU forecasting
   * - ``FCST_8k_NPU``
     - ~8,000
     - Large NPU forecasting
   * - ``FCST_20k_NPU``
     - ~20,000
     - Maximum NPU forecasting

Image Classification Models
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Model Name
     - Parameters
     - Description
   * - ``MobileNetV2_Tiny``
     - ~10,000
     - Minimal image model
   * - ``MobileNetV2_Small``
     - ~50,000
     - Small image model
   * - ``MobileNetV2_Medium``
     - ~100,000
     - Medium image model
   * - ``CustomCNN_Small``
     - ~20,000
     - Simple custom CNN

Model Selection Guide
---------------------

**By Device Type:**

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Device Class
     - Recommended Models
     - Max Size
   * - Entry-level M0+
     - ``*_100``, ``*_500``
     - ~1k params
   * - Mid-range
     - ``*_1k``, ``*_2k``
     - ~4k params
   * - High-performance
     - ``*_4k``, ``*_6k``
     - ~13k params
   * - NPU devices
     - ``*_NPU`` variants
     - ~55k params

**By Task Complexity:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Task Complexity
     - Classes/Output
     - Recommended Size
   * - Simple (2 classes)
     - Binary
     - 100-500 params
   * - Moderate (3-5 classes)
     - Few classes
     - 1k-2k params
   * - Complex (6+ classes)
     - Many classes
     - 4k+ params
   * - Very complex
     - High accuracy needed
     - 6k-13k params

Model Architecture Details
--------------------------

**Classification Architecture (CLS_*k_NPU):**

.. code-block:: text

   Input (1, 512, 1)
   ├── Conv1: 1→4 ch, kernel=5
   ├── BN + ReLU
   ├── MaxPool: 2x1
   ├── Conv2: 4→8 ch, kernel=5
   ├── BN + ReLU
   ├── MaxPool: 2x1
   ├── Conv3: 8→16 ch, kernel=3
   ├── BN + ReLU
   ├── Flatten
   └── FC: → num_classes

**Anomaly Detection Architecture (AD_*k_NPU):**

.. code-block:: text

   Encoder:
   ├── Conv1: 1→4 ch
   ├── Conv2: 4→8 ch
   └── Conv3: 8→bottleneck

   Decoder:
   ├── ConvT1: bottleneck→8 ch
   ├── ConvT2: 8→4 ch
   └── ConvT3: 4→1 ch

Using Models
------------

**In Configuration:**

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'

**Listing Available Models:**

.. code-block:: python

   from tinyml_tinyverse.common.models import MODEL_REGISTRY
   print(list(MODEL_REGISTRY.keys()))

**Model Information:**

.. code-block:: python

   from tinyml_tinyverse.common.models import MODEL_REGISTRY

   model_class = MODEL_REGISTRY['CLS_4k_NPU']
   model = model_class(config={}, input_features=512, variables=1, num_classes=2)
   params = sum(p.numel() for p in model.parameters())
   print(f"Parameters: {params}")
