======================
System Architecture
======================

This page describes the architecture of Tiny ML Tensorlab and how its
components work together to provide an end-to-end ML workflow.

High-Level Architecture
-----------------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                        USER INPUT                               │
   │                    (YAML Config File)                           │
   └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     tinyml-modelzoo                             │
   │  • Entry point for users                                        │
   │  • Example configurations                                       │
   │  • Model definitions                                            │
   └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    tinyml-modelmaker                            │
   │  • Orchestrates the workflow                                    │
   │  • Calls training, testing, compilation                         │
   └─────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ tinyml-tinyverse │ │   tinyml-model   │ │  TI Neural       │
   │  • Training      │ │   optimization   │ │  Network         │
   │  • Feature ext.  │ │  • Quantization  │ │  Compiler        │
   │  • Testing       │ │  • QAT/PTQ       │ │  • Device code   │
   └──────────────────┘ └──────────────────┘ └──────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                      OUTPUT ARTIFACTS                           │
   │  • mod.a (compiled model)                                       │
   │  • tvmgen_default.h (API header)                                │
   │  • test_vector.c (validation data)                              │
   └─────────────────────────────────────────────────────────────────┘

Component Details
-----------------

tinyml-modelzoo
~~~~~~~~~~~~~~~

**Purpose**: Customer-facing entry point containing models and examples.

**Key Contents**:

* ``examples/`` - Ready-to-run YAML configurations for various applications
* ``tinyml_modelzoo/models/`` - Neural network model definitions
* ``tinyml_modelzoo/model_descriptions/`` - Model metadata for GUI integration
* ``tinyml_modelzoo/device_info/`` - Device performance benchmarks

**Model Organization**:

.. code-block:: text

   tinyml_modelzoo/models/
   ├── classification.py      # Time series classification models
   ├── regression.py          # Time series regression models
   ├── anomalydetection.py    # Autoencoder models
   ├── forecasting.py         # Time series forecasting models
   └── image.py               # Image classification models

tinyml-modelmaker
~~~~~~~~~~~~~~~~~

**Purpose**: Orchestrates the end-to-end workflow.

**Key Functions**:

* Parse configuration files
* Manage project directory structure
* Call training scripts from tinyml-tinyverse
* Invoke quantization from tinyml-modeloptimization
* Run TI Neural Network Compiler

**Output Directory Structure**:

.. code-block:: text

   data/projects/<dataset_name>/run/<run_name>/
   ├── training/
   │   ├── base/              # Float32 model training outputs
   │   │   ├── best_model.pt
   │   │   ├── training_log.csv
   │   │   └── *.png (visualizations)
   │   └── quantization/      # Quantized model outputs
   │       ├── best_model_quantized.pt
   │       ├── best_model.onnx
   │       └── golden_vectors/
   ├── testing/               # Test results
   └── compilation/           # Compiled artifacts
       └── artifacts/
           ├── mod.a
           └── tvmgen_default.h

tinyml-tinyverse
~~~~~~~~~~~~~~~~

**Purpose**: Core training infrastructure and utilities.

**Key Components**:

* ``references/`` - Task-specific training scripts:

  * ``timeseries_classification/train.py``
  * ``timeseries_regression/train.py``
  * ``timeseries_forecasting/train.py``
  * ``timeseries_anomalydetection/train.py``
  * ``image_classification/train.py``

* ``common/`` - Shared utilities:

  * ``datasets/`` - Dataset loaders for various formats
  * ``models/`` - Base model classes and utilities
  * ``transforms/`` - Feature extraction transforms
  * ``augmenters/`` - Data augmentation functions
  * ``compilation/`` - TVM compilation interface

tinyml-modeloptimization
~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Model quantization toolkit.

**Key Features**:

* **TINPU Wrappers** - For NPU-enabled devices (F28P55, etc.)
* **Generic Wrappers** - For non-NPU devices
* **QAT Support** - Quantization-aware training
* **PTQ Support** - Post-training quantization

Data Flow
---------

**Training Pipeline**:

.. code-block:: text

   Raw Data → Data Loading → Augmentation → Feature Extraction → Model → Loss → Optimizer
                                                                  ↑
                                                          (model.pt saved)

**Inference Pipeline**:

.. code-block:: text

   Raw Data → Feature Extraction → Quantized Model → Output Classes/Values
                 ↑                       ↑
       (user_input_config.h)       (mod.a from compilation)

Configuration System
--------------------

The entire workflow is driven by YAML configuration files:

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'my_dataset'
     input_data_path: '/path/to/data'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'

   training:
     model_name: 'CLS_1k_NPU'
     batch_size: 256
     training_epochs: 20

   compilation:
     enable: True

See :doc:`/getting_started/understanding_config` for complete configuration reference.

Integration Points
------------------

**External Tools**:

* **TI Neural Network Compiler (NNC)** - Compiles ONNX models to device code
* **Code Composer Studio (CCS)** - IDE for device deployment
* **C2000Ware / MSPM0 SDK** - Device-specific libraries

**Data Formats**:

* **Input**: CSV, TXT, NPY, PKL
* **Model**: PyTorch (.pt) → ONNX (.onnx)
* **Output**: Static library (.a), C header (.h)
