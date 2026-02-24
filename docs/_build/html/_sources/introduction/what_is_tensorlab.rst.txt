==========================
What is Tiny ML Tensorlab?
==========================

Tiny ML Tensorlab is Texas Instruments' comprehensive AI toolchain for developing,
training, optimizing, and deploying machine learning models on resource-constrained
microcontrollers (MCUs).

Overview
--------

The toolchain provides an end-to-end workflow that takes you from raw data to
a compiled model running on a TI MCU. It handles:

* **Data Preparation** - Format and preprocess datasets for training
* **Feature Extraction** - Transform raw signals into meaningful features
* **Model Training** - Train neural networks optimized for embedded deployment
* **Model Optimization** - Quantize models for efficient inference (2/4/8-bit)
* **Compilation** - Generate device-specific code using TI's Neural Network Compiler
* **Analysis** - Evaluate model performance before deployment

Target Applications
-------------------

Tiny ML Tensorlab is designed for industrial and IoT applications where you need
to run AI inference directly on the microcontroller:

**Predictive Maintenance**
   * Motor bearing fault detection
   * Equipment anomaly monitoring
   * Vibration analysis

**Power Systems**
   * DC and AC arc fault detection
   * Grid stability prediction
   * Power quality monitoring

**Control Systems**
   * Torque estimation
   * Temperature forecasting
   * Load prediction

**Sensor Processing**
   * Activity recognition
   * Gas classification
   * Motion detection

Key Capabilities
----------------

**Supported ML Tasks**

.. list-table::
   :widths: 30 70

   * - Time Series Classification
     - Categorize sensor signals into discrete classes
   * - Time Series Regression
     - Predict continuous values from sensor inputs
   * - Time Series Forecasting
     - Predict future values based on historical patterns
   * - Anomaly Detection
     - Identify abnormal patterns using autoencoders
   * - Image Classification
     - Categorize images into classes

**Device Support**

Over 20 TI microcontrollers are supported, including:

* **C2000 DSP family** - F28P55 (with NPU), F28P65, F2837, F28003, F28004
* **MSPM0 family** - MSPM0G3507, MSPM0G5187 (with NPU)
* **MSPM33C family** - MSPM33C32, MSPM33C34
* **AM13 family** - AM13E2 (with NPU)
* **AM26x family** - AM263, AM263P, AM261
* **Connectivity devices** - CC2755, CC1352

**Model Optimization**

* Quantization-Aware Training (QAT) for best accuracy
* Post-Training Quantization (PTQ) for fast deployment
* Support for 2-bit, 4-bit, and 8-bit weight quantization

**NPU Acceleration**

Select devices (F28P55, AM13E2, MSPM0G5187) include TI's Neural Processing Unit
(TINPU) for hardware-accelerated inference, providing faster execution and
lower power consumption.

Repository Structure
--------------------

Tiny ML Tensorlab consists of four main components:

**tinyml-modelzoo** (Customer Entry Point)
   Contains model definitions, example configurations, and pre-trained checkpoints.
   This is where you start when using the toolchain.

   * ``examples/`` - Ready-to-run configuration files
   * ``tinyml_modelzoo/models/`` - Model architectures
   * ``docs/`` - NPU configuration guidelines

**tinyml-modelmaker**
   Orchestrates the end-to-end workflow. Provides scripts to run training,
   testing, and compilation.

   * ``examples/`` - Additional example configurations
   * ``docs/`` - Detailed documentation and guides
   * ``scripts/`` - Utility scripts

**tinyml-tinyverse**
   Core training infrastructure. Contains:

   * Dataset classes for various data formats
   * Training scripts for all task types
   * Feature extraction transforms
   * Common utilities

**tinyml-modeloptimization**
   Quantization toolkit with:

   * TINPU wrappers for NPU deployment
   * Generic quantization wrappers
   * QAT and PTQ implementations

Workflow Summary
----------------

A typical workflow with Tiny ML Tensorlab:

1. **Prepare Your Data**
   Format your dataset according to the required structure (see :doc:`/byod/index`)

2. **Create a Configuration File**
   Define your training parameters in a YAML config (see :doc:`/getting_started/understanding_config`)

3. **Run Training**

   .. code-block:: bash

      ./run_tinyml_modelzoo.sh examples/your_config/config.yaml

4. **Review Results**
   Examine training metrics, visualizations, and analysis outputs

5. **Deploy to Device**
   Copy compiled artifacts to your CCS project and flash to the MCU

Next Steps
----------

* :doc:`architecture` - Learn about the system architecture
* :doc:`/installation/index` - Set up your development environment
* :doc:`/getting_started/quickstart` - Train your first model
