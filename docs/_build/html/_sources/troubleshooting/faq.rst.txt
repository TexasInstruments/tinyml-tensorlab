====
FAQ
====

Frequently asked questions about Tiny ML Tensorlab.

General Questions
-----------------

**What is Tiny ML Tensorlab?**

Tiny ML Tensorlab is Texas Instruments' MCU AI Toolchain for developing,
training, and deploying machine learning models on TI microcontrollers.
It provides end-to-end support from data processing through device deployment.

**What devices are supported?**

Over 20 TI microcontrollers across multiple families:

* C2000 (F28P55, F28P65, F2837, etc.)
* MSPM0 (MSPM0G3507, MSPM0G3519, MSPM0G5187)
* MSPM33C (MSPM33C32, MSPM33C34)
* AM13 (AM13E2)
* AM26x (AM263, AM263P, AM261)
* Connectivity (CC2755, CC1352, CC1354, CC35X1)

See :doc:`/devices/device_overview` for the complete list.

**Which devices have NPU?**

Three devices include the TINPU neural accelerator:

* F28P55 (C2000 family)
* AM13E2 (AM13 family)
* MSPM0G5187 (MSPM0 family)

**Is GPU required for training?**

No, GPU is optional. CPU training works well for most models.
GPU accelerates training for larger models and NAS.

**What operating systems are supported?**

* Linux (Ubuntu 20.04/22.04 recommended)
* Windows 10/11 (native or WSL2)

macOS is not officially supported but may work with some limitations.

Installation Questions
----------------------

**Why is Python 3.10 required?**

Tiny ML Tensorlab depends on specific package versions that are tested
and validated with Python 3.10. Other versions may have compatibility
issues with the TI neural network compiler.

**Can I use conda instead of venv?**

Yes, conda works. Create an environment with Python 3.10:

.. code-block:: bash

   conda create -n tinyml python=3.10
   conda activate tinyml

**Do I need to install TI compilers?**

TI compilers are only needed for compilation. For training and testing,
they are not required. Install them when you're ready to deploy.

Dataset Questions
-----------------

**What data formats are supported?**

CSV files are the primary format. Each sample is a CSV file with:

* Time series data in columns
* No header row (unless configured)
* Numeric values only

**How much training data do I need?**

Guidelines:

* Minimum: 100 samples per class
* Recommended: 500+ samples per class
* More data generally helps, especially for complex tasks

**Can I use my own dataset?**

Yes! See the :doc:`/byod/index` section for dataset format requirements.

**How should I split training/test data?**

Tiny ML Tensorlab can automatically split:

.. code-block:: yaml

   dataset:
     data_split_type: 'random'
     data_split_ratio: [0.8, 0.1, 0.1]  # train, val, test

Model Questions
---------------

**How do I choose a model size?**

Consider:

* Device memory constraints
* Accuracy requirements
* Inference latency needs

General guidelines:

* Simple tasks: 100-500 parameters
* Medium tasks: 1k-4k parameters
* Complex tasks: 4k-13k parameters

**What's the difference between NPU and non-NPU models?**

NPU models (ending in ``_NPU``) are designed for hardware acceleration:

* Follow strict architectural constraints
* Faster inference on NPU devices
* May have slightly different accuracy

**Can I use pre-trained models?**

Yes, via the BYOM (Bring Your Own Model) feature.
See :doc:`/byom/compilation_only`.

**How do I improve model accuracy?**

Options:

1. More training data
2. Larger model
3. Better feature extraction
4. Data augmentation
5. More training epochs

Training Questions
------------------

**How long does training take?**

Depends on model size and data:

* Hello World example: 2-5 minutes
* Typical classification: 10-30 minutes
* Large models with NAS: Hours

**Why is my accuracy low?**

Check:

1. Data quality (use GoF test)
2. Sufficient training data
3. Appropriate feature extraction
4. Correct data labeling

**What if training doesn't converge?**

Try:

1. Lower learning rate
2. Different model architecture
3. Check data for issues
4. Increase training epochs

**How do I handle class imbalance?**

Options:

* Oversample minority classes
* Undersample majority classes
* Use class weights in training
* Collect more data for minority classes

Deployment Questions
--------------------

**What IDE do I need?**

Code Composer Studio (CCS) is TI's official IDE.
Download from: https://www.ti.com/tool/CCSTUDIO

**How do I get the compiled model onto my device?**

1. Import CCS project or add artifacts to existing project
2. Build the project
3. Flash to device using CCS debugger
4. Run and debug

**What's the inference latency?**

Varies by model and device:

* NPU devices: 100-600 µs typical
* CPU devices: 500 µs - 10 ms typical

See :doc:`/devices/device_overview` for estimates.

**Can I run multiple models?**

Yes, but consider memory constraints. Each model needs:

* Flash for weights
* RAM for activations and buffers

Advanced Questions
------------------

**How does quantization work?**

Quantization reduces precision from 32-bit float to 8-bit (or lower) integers:

* Reduces model size 4x (float32 → int8)
* Speeds up inference
* Usually <1% accuracy loss

**What is NAS?**

Neural Architecture Search automatically finds optimal model architectures
for your specific task and constraints.

**Can I customize the training pipeline?**

Yes, you can add custom models to ``tinyml-modelzoo``.
See :doc:`/byom/adding_models` for step-by-step instructions.

**Is there an API for integration?**

The MLBackend provides REST API endpoints for:

* Training management
* Dataset operations
* Model compilation

Support Questions
-----------------

**Where do I report bugs?**

GitHub Issues: https://github.com/TexasInstruments/tinyml-tensorlab/issues

**How do I get help?**

1. Check this documentation
2. Search GitHub Issues
3. TI E2E forums for device-specific questions
4. Create a GitHub Issue

**Are there example projects?**

Yes! See ``tinyml-modelzoo/examples/`` for ready-to-run examples:

* Arc fault detection
* Motor bearing fault
* Hello World
* And more

**How do I stay updated?**

* Watch the GitHub repository
* Check TI's Edge AI page
* Follow TI on social media

Still Have Questions?
---------------------

If your question isn't answered here:

1. Check :doc:`common_errors` for error-specific help
2. Browse the full documentation
3. Search GitHub Issues
4. Create a new Issue with your question
