=============
Common Errors
=============

This page lists common errors encountered when using Tiny ML Tensorlab
and their solutions.

Installation Errors
-------------------

**Python Version Error**

.. code-block:: text

   ERROR: Package requires Python 3.10.*

**Solution:**

Use Python 3.10.x specifically:

.. code-block:: bash

   pyenv install 3.10.13
   pyenv local 3.10.13

**Package Conflict**

.. code-block:: text

   ERROR: Cannot install package due to conflicting dependencies

**Solution:**

Use a fresh virtual environment:

.. code-block:: bash

   python -m venv venv_fresh
   source venv_fresh/bin/activate
   pip install -e .

**TI Compiler Not Found**

.. code-block:: text

   ERROR: ti_cgt_c2000 not found

**Solution:**

Set environment variable:

.. code-block:: bash

   export TOOLS_PATH=$HOME/ti

See :doc:`/installation/environment_variables` for details.

Dataset Errors
--------------

**Dataset Not Found**

.. code-block:: text

   ERROR: Dataset 'my_dataset' not found

**Solution:**

Check dataset path and name:

.. code-block:: yaml

   dataset:
     dataset_name: 'my_dataset'
     input_data_path: '/full/path/to/my_dataset'  # Absolute path

**Invalid Dataset Format**

.. code-block:: text

   ERROR: Could not parse annotations.yaml

**Solution:**

Verify annotations.yaml format:

.. code-block:: yaml

   name: my_dataset
   description: My dataset description
   task_type: classification

See :doc:`/byod/classification_format` for correct format.

**Missing Classes Directory**

.. code-block:: text

   ERROR: 'classes' directory not found in dataset

**Solution:**

Verify directory structure:

.. code-block:: text

   my_dataset/
   ├── annotations.yaml
   └── classes/           # Must be named 'classes'
       ├── class_a/
       └── class_b/

**Empty CSV Files**

.. code-block:: text

   WARNING: Empty CSV file detected: sample.csv

**Solution:**

Check CSV files have valid data. Minimum requirements:

* At least one column
* At least one row of data
* No empty files

Training Errors
---------------

**Out of Memory (OOM)**

.. code-block:: text

   RuntimeError: CUDA out of memory

**Solution:**

Reduce batch size:

.. code-block:: yaml

   training:
     batch_size: 64  # Reduce from 256

Or use CPU:

.. code-block:: yaml

   training:
     num_gpus: 0

**Model Not Found**

.. code-block:: text

   KeyError: 'CLS_4k_NPU' not found in model registry

**Solution:**

Check model name spelling. List available models:

.. code-block:: bash

   python -c "from tinyml_modelzoo.models import list_models; print(list_models())"

**Feature Extraction Preset Not Found**

.. code-block:: text

   ERROR: Feature extraction preset 'My_Preset' not found

**Solution:**

Use a valid preset name:

.. code-block:: yaml

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'

**Training Loss Not Decreasing**

.. code-block:: text

   Epoch 20: Loss 2.45 (same as epoch 1)

**Solutions:**

1. Check learning rate:

   .. code-block:: yaml

      training:
        learning_rate: 0.001  # Try different values

2. Check data preprocessing
3. Verify labels are correct
4. Use GoF test to check class separability

**NaN Loss**

.. code-block:: text

   Epoch 5: Loss = nan

**Solutions:**

1. Lower learning rate:

   .. code-block:: yaml

      training:
        learning_rate: 0.0001

2. Add gradient clipping
3. Check for data issues (inf, nan values)

Compilation Errors
------------------

**NPU Constraint Violation**

.. code-block:: text

   ERROR: Channel count 5 not a multiple of 4

**Solution:**

Use NPU-compatible model:

.. code-block:: yaml

   training:
     model_name: 'CLS_4k_NPU'  # _NPU suffix models

**Kernel Size Exceeds Limit**

.. code-block:: text

   ERROR: GCONV kernel height 8 exceeds maximum 7

**Solution:**

Use appropriate model or modify architecture. See :doc:`/devices/npu_guidelines`.

**Unsupported Layer Type**

.. code-block:: text

   ERROR: Layer type 'LSTM' not supported for target device

**Solution:**

Use only supported layers (Conv, Pool, FC, BN, ReLU).

**Model Too Large**

.. code-block:: text

   ERROR: Model size 80KB exceeds device Flash 64KB

**Solution:**

Use smaller model:

.. code-block:: yaml

   training:
     model_name: 'CLS_1k_NPU'  # Smaller model

Quantization Errors
-------------------

**Calibration Data Missing**

.. code-block:: text

   ERROR: PTQ requires calibration data

**Solution:**

Provide calibration data:

.. code-block:: yaml

   training:
     quantization: 2
     quantization_method: 'PTQ'
     quantization_weight_bitwidth: 8
     quantization_activation_bitwidth: 8

**Quantization Accuracy Drop**

.. code-block:: text

   Float accuracy: 98%, INT8 accuracy: 75%

**Solutions:**

1. Use QAT instead of PTQ:

   .. code-block:: yaml

      training:
        quantization: 2
        quantization_method: 'QAT'
        quantization_weight_bitwidth: 8
        quantization_activation_bitwidth: 8

2. Try higher bit widths (8-bit instead of 4-bit or 2-bit)
3. Use a larger model

Deployment Errors
-----------------

**CCS Import Fails**

.. code-block:: text

   Error importing project: SDK not found

**Solution:**

Install required SDK:

* C2000WARE for C2000 devices
* MSPM0 SDK for MSPM0 devices

**Linker Error: Undefined Symbol**

.. code-block:: text

   undefined symbol: mod_inference

**Solution:**

Add mod.a to linker settings:

1. Project Properties → Build → Linker → File Search Path
2. Add library: ``${PROJECT_ROOT}/model/mod.a``

**Runtime Hard Fault**

.. code-block:: text

   Exception: Hard Fault at 0x00001234

**Solutions:**

1. Check memory alignment
2. Verify buffer sizes
3. Check stack size
4. Enable debug symbols

**Wrong Inference Results**

.. code-block:: text

   Expected class 0, got class 2 (all inputs)

**Solutions:**

1. Verify input data format matches training
2. Check preprocessing (normalization)
3. Use test vectors to validate
4. Compare with Python inference

Configuration Errors
--------------------

**YAML Syntax Error**

.. code-block:: text

   yaml.scanner.ScannerError: mapping values are not allowed

**Solution:**

Check YAML syntax:

* Proper indentation (spaces, not tabs)
* Colons followed by space
* Quotes around special characters

**Missing Required Field**

.. code-block:: text

   KeyError: 'target_device' is required

**Solution:**

Add required field:

.. code-block:: yaml

   common:
     target_device: 'F28P55'
     task_type: 'generic_timeseries_classification'

**Invalid Device Name**

.. code-block:: text

   ERROR: Device 'F28P55X' not supported

**Solution:**

Check device name spelling. Valid names:

* F28P55, F28P65, F29H85, F29P58, F29P32
* F2837, F28004, F28003, F280013, F280015
* MSPM0G3507, MSPM0G3519, MSPM0G5187
* AM263, AM263P, AM261, AM13E2
* CC2755, CC1352, CC1354, CC35X1

Getting Help
------------

If your error isn't listed:

1. Check the full error traceback
2. Search existing GitHub issues
3. Create new issue with:

   * Full error message
   * Configuration file
   * Steps to reproduce
   * Environment details

Report issues at: https://github.com/TexasInstruments/tinyml-tensorlab/issues
