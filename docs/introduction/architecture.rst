===================
System Architecture
===================

How the components of Tiny ML ModelZoo work together.

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
   │  • Entry point (run_tinyml_modelzoo.sh / .bat / .ps1)           │
   │  • Example configurations                                       │
   │  • Model definitions and descriptions                           │
   └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    tinyml_modelmaker                            │
   │  • Orchestrates the workflow                                    │
   │  • Calls training, testing, compilation                         │
   └─────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ tinyml_tinyverse │ │ tinyml_torch     │ │  TI Neural       │
   │  • Training      │ │ modelopt         │ │  Network         │
   │  • Feature ext.  │ │  • Quantization  │ │  Compiler        │
   │  • Testing       │ │  • QAT/PTQ       │ │  (ti_mcu_nnc)    │
   └──────────────────┘ └──────────────────┘ └──────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                      OUTPUT ARTIFACTS                           │
   │  • mod.a (compiled model)                                       │
   │  • tvmgen_default.h (API header)                                │
   │  • test_vector.c, user_input_config.h (validation data)         │
   └─────────────────────────────────────────────────────────────────┘

Packages and How They Are Delivered
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Package
     - Role
     - How you get it
   * - ``tinyml_modelzoo``
     - Models, model descriptions, examples, run scripts
     - The wheel, or a clone of ``tinyml-modelzoo``
   * - ``tinyml_modelmaker``
     - Workflow orchestration
     - Dependency of ``tinyml_modelzoo`` (prebuilt wheel)
   * - ``tinyml_tinyverse``
     - Training, datasets, feature extraction
     - Dependency of ModelMaker (prebuilt wheel)
   * - ``tinyml_torchmodelopt``
     - Quantization
     - Dependency of ModelMaker (prebuilt wheel)
   * - ``ti_mcu_nnc``
     - TI Neural Network Compiler (TVM based)
     - Dependency of ModelMaker (Python 3.10 only)

Only ``tinyml-modelzoo`` is cloned, and only if you want to add or change models
(:doc:`/installation/index`).

Component Details
-----------------

tinyml-modelzoo
~~~~~~~~~~~~~~~

**Purpose**: the user-facing entry point with models and examples.

* ``examples/``: ready-to-run YAML configurations
* ``tinyml_modelzoo/models/``: neural network model definitions
* ``tinyml_modelzoo/model_descriptions/``: model metadata (the ``model_name`` values used in configs, supported devices)
* ``docs/``: guides (config YAML, bring your own model, NPU guidelines)

.. code-block:: text

   tinyml_modelzoo/models/
   ├── classification.py      # time series classification
   ├── regression.py          # time series regression
   ├── forecasting.py         # time series forecasting
   ├── anomalydetection.py    # autoencoder models
   ├── audio.py               # audio classification
   ├── image.py               # image classification
   ├── radar.py               # radar point cloud classification
   ├── feature_extraction.py  # feature extraction models
   └── base.py                # base classes

tinyml_modelmaker
~~~~~~~~~~~~~~~~~

**Purpose**: orchestrates the end-to-end workflow.

* Parses the config and applies defaults
* Manages the project directory
* Calls the training scripts of ``tinyml_tinyverse``
* Invokes quantization from ``tinyml_torchmodelopt``
* Runs the TI Neural Network Compiler
* Its task modules are ``timeseries``, ``audio``, ``vision`` and ``radar``

Output folder layout: see :doc:`/getting_started/understanding_run_output`.

tinyml_tinyverse
~~~~~~~~~~~~~~~~

**Purpose**: core training infrastructure.

* ``references/``: task-specific training scripts
* ``common/datasets``: dataset loaders
* ``common/models``: base model classes
* ``common/transforms``: feature extraction transforms
* ``common/augmenters``: data augmentation

tinyml_torchmodelopt
~~~~~~~~~~~~~~~~~~~~

**Purpose**: model quantization toolkit.

* **TINPU wrappers** for NPU-enabled devices
* **Generic wrappers** for devices without an NPU
* **QAT** and **PTQ** support, plus automatic mixed-precision quantization

Which Install Do I Need?
------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 50 38

   * - Mode
     - Use case
     - Install
   * - BYOD
     - Bring your own data, use TI models as-is or change model parameters in the config
     - Wheel install (config only)
   * - BYOM
     - Add or modify model architectures and train end-to-end
     - Either install with ``training.custom_model_path``, or the clone install to edit ``tinyml_modelzoo/models``
   * - BYOM
     - Already have a trained ONNX model and need to compile it for a TI device
     - See :doc:`/BRING_YOUR_OWN_MODEL`

.. note::

   **BYOD** is Bring Your Own Data and **BYOM** is Bring Your Own Model. Most users only
   edit YAML configs. See :doc:`/getting_started/understanding_config`.

Data Flow
---------

**Training pipeline**:

.. code-block:: text

   (Raw Data + Config) → Data Loading → Augmentation → Feature Extraction → Float Training → Quantization → Compilation
                                                              │                  │               │             │
                                                              ▼                  ▼               ▼             ▼
                                                    user_input_config.h     model.onnx      model.onnx     artifacts/
                                                                              (float)       (quantized)    ├── mod.a
                                                                                                           └── tvmgen_default.h

**Inference pipeline on the device**:

.. code-block:: text

   Raw Data → Feature Extraction → Quantized Model → Output Classes/Values
                 ↑                       ↑
       (user_input_config.h)       (mod.a from compilation)

Configuration System
--------------------

The workflow is driven by one YAML file:

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: my_dataset
     input_data_path: /path/to/data
   data_processing_feature_extraction:
     feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
   training:
     model_name: CLS_1k_NPU
     batch_size: 256
     training_epochs: 20
   compilation:
     enable: True

See :doc:`/getting_started/understanding_config` and :doc:`/CONFIG_YAML_GUIDE`.

Integration Points
------------------

* **TI Neural Network Compiler** compiles ONNX models to device code
* **Compiler tools** (C2000 CGT, TI Arm Clang, C29 Clang): :doc:`/installation/environment_variables`
* **Code Composer Studio and device SDKs**: :doc:`/DEPLOYING_TO_SDK_EXAMPLE`
* **Data formats**: input CSV, TXT, NPY, PKL; model PyTorch to ONNX; output static library (``.a``) and C header (``.h``)
