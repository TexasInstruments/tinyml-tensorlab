=========================
What is Tiny ML ModelZoo?
=========================

Tiny ML ModelZoo is Texas Instruments' central repository of AI models, example
applications and configurations for microcontroller (MCU) applications. Together with the
toolchain it installs, it takes you from raw data to a compiled model running on a TI MCU.

.. note::

   Up to release 1.4.0 this toolchain was distributed as *Tiny ML Tensorlab*
   (``tinyml-tensorlab``), a set of repositories you cloned together. From 1.5.0,
   ``tinyml-modelzoo`` is the entry point and everything else comes in as prebuilt wheels
   (:doc:`/installation/index`). ``tinyml-tensorlab`` is deprecated and its source is private.

Overview
--------

The toolchain handles:

* **Data preparation**: format and preprocess datasets for training
* **Feature extraction**: transform raw signals into meaningful features
* **Model training**: train neural networks designed for embedded deployment
* **Model optimization**: quantize models for efficient inference
* **Compilation**: generate device-specific code with TI's Neural Network Compiler
* **Analysis**: evaluate model performance before deployment

Target Applications
-------------------

**Predictive maintenance**
   Motor bearing, gearbox and fan blade fault detection, equipment anomaly monitoring, vibration analysis.

**Power systems**
   DC and AC arc fault detection, grid fault and stability, electrical fault classification.

**Control systems**
   Torque estimation, motor speed prediction, temperature forecasting, battery state-of-charge estimation.

**Sensor processing and sensing**
   Activity and fall detection, PIR and Wi-Fi CSI presence detection, gas classification,
   radar point cloud and surface classification.

**Audio and vision**
   Keyword and wake word detection, cough and glass break detection, image classification.

Key Capabilities
----------------

**Supported ML tasks**

.. list-table::
   :widths: 30 70

   * - Time Series Classification
     - Categorize sensor signals into discrete classes
   * - Time Series Regression
     - Predict continuous values from sensor inputs
   * - Time Series Forecasting
     - Predict future values from historical patterns
   * - Time Series Anomaly Detection
     - Identify abnormal patterns using autoencoders
   * - Audio Classification
     - Classify audio from MFCC features
   * - Image Classification
     - Categorize images into classes
   * - Radar Classification
     - Classify point-cloud frames or FFT range profiles from radar sensors

**Device support**

33 MCU and wireless devices (as listed in the README of the 1.5.0 release) across these
families. See :doc:`/devices/index` for details.

* **C2000 (C28 DSP)**: F28P55 (with NPU), F28P65, F2837, F2838x, F28003, F28004 and others
* **C2000 (C29 DSP)**: F29H85, F29P58, F29P32
* **MSPM0**: MSPM0G3507, MSPM0G3519, MSPM0G5187 (with NPU)
* **MSPM33C**: MSPM33C32, MSPM33C34
* **AM13**: AM13E2 (with NPU)
* **AM26x**: AM263, AM263P, AM261
* **Connectivity**: CC2755, CC2745, CC1312, CC1314, CC1352, CC1354, CC35X1
* **Radar**: xWRL6432

**Example applications and models**

The ``examples/`` folder ships ready-to-run applications (the generic ones for each task
plus dedicated applications such as arc fault, motor bearing fault, ECG, wake word and
Wi-Fi presence detection), and the model zoo contains 85 models. See
:doc:`/examples/index` and :doc:`/appendix/model_zoo_reference`.

**Model optimization**

* Quantization-aware training (QAT) and post-training quantization (PTQ)
* Hessian-aware (HAWQ) automatic mixed-precision quantization
* Residual connection support for NPU quantization

**NPU acceleration**

Select devices (F28P55, MSPM0G5187, AM13E2) include TI's TinyEngine NPU for
hardware-accelerated inference with faster execution and lower power. See
:doc:`/NPU_CONFIGURATION_GUIDELINES`.

What's New in 1.6.0
-------------------

* Radar surface classification example (``radar_surface_classification``) and two new radar models, ``RF_MLP_3L_11K`` and ``RF_DualStream_PointCloud_19K``
* ``radar_pose_and_fall_detection`` renamed to ``radar_posture_and_fall_detection``, with a new radar processing pipeline guide
* 85 models (was 83)

What's New in 1.5.0
-------------------

* Standalone pip install of ``tinyml-modelzoo``; one repository to clone for model work
* New tasks and devices: audio classification, radar point cloud classification, AM13E2 vision
* Early stopping on by default, NaN-loss detection, memory estimation before training
* macOS Apple Silicon (MPS) support, Python 3.14 / PyTorch 2.11 compatibility for training
* Integer on-device training, learned filterbank front end, new example applications

See the release history in the README and the changelog in the appendix.

Repository Structure
--------------------

.. code-block:: text

   tinyml-modelzoo/
   ├── examples/               # ready-to-run example configurations
   ├── tinyml_modelzoo/
   │   ├── models/             # neural network model definitions
   │   └── model_descriptions/ # model metadata (names used in configs)
   ├── docs/                   # guides
   ├── run_tinyml_modelzoo.sh / .bat / .ps1
   └── ADDING_NEW_MODELS.md

The other parts of the toolchain are installed as packages, not cloned:

``tinyml_modelmaker``
   Orchestrates the workflow: reads the config, runs training, testing and compilation.

``tinyml_tinyverse``
   Core training infrastructure: datasets, training scripts, feature extraction transforms.

``tinyml_torchmodelopt``
   Quantization toolkit (TINPU and generic wrappers, QAT/PTQ).

Workflow Summary
----------------

1. **Install**: :doc:`/installation/index`
2. **Prepare data and a config**: :doc:`/byod/index`, :doc:`/CONFIG_YAML_GUIDE`
3. **Run**:

   .. code-block:: bash

      ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

4. **Review results**: :doc:`/getting_started/understanding_run_output`
5. **Deploy to a device**: :doc:`/DEPLOYING_TO_SDK_EXAMPLE`

Next Steps
----------

* :doc:`architecture`
* :doc:`/installation/index`
* :doc:`/getting_started/quickstart`
