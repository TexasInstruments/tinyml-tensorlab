================================================
Tiny ML Agentic Skill — AI-Guided Model Creation
================================================

.. warning::
   **Beta Feature** — This skill is experimental. Workflow and output may change.

What Is This Skill?
====================

The **Tiny ML Agentic Skill** is an end-to-end AI assistant that guides you through
the complete machine learning workflow for embedded devices. It handles everything
from project configuration, data analysis, model selection, training, compilation,
and deployment to TI microcontrollers — all conversationally.

Instead of manually juggling configuration files, dataset preparation, feature
extraction, model selection, quantization modes, and compilation settings, you
simply answer guided questions and Claude automates the technical details.

**Example:** "I have ECG sensor data and want to detect abnormal patterns on an
F28P55."
→ The skill interviews you, analyzes your data, recommends appropriate feature
transforms and models, automatically configures all 50+ parameters, trains your
model, compiles optimized binaries, and generates a Code Composer Studio project
ready for device deployment.

Prerequisites
=============

**Required:**

* **tinyml-tensorlab** — Installed and configured on your system

  * `Download here <https://github.com/TexasInstruments/tinyml-tensorlab>`__
  * GPU strongly recommended for faster training (CPU works but is slower)

* **Claude Code** — The CLI tool or IDE extension where this skill runs

**Optional:**

* **Your Dataset** — CSV, TXT, or NumPy format with pre-processed, cleaned sensor or image data

  * Sample datasets are included in ``tinyml-tensorlab/tinyml-modelzoo/examples/`` for testing

* **Code Composer Studio (TI CCStudio IDE)** — Needed for on-device deployment

  * `Download here <https://www.ti.com/tool/CCSTUDIO>`__ (free)
  * Skip if you only want to train and test on your PC

**Data Requirements:**
You must provide pre-processed, cleaned data — the skill does not clean or preprocess raw datasets.

Installation
============

Step 1: Install tinyml-tensorlab
--------------------------------

Clone and set up the main repository as explained in prior sections:

.. code-block:: bash

   git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab

Step 2: Activate Claude Code Skill
-----------------------------------

To install and begin using the tinyml-agentic-skill, follow the steps given below:
1. Open claude code CLI and register tinyml-agent-skills as a Claude Code Plugin marketplace by running the following command in Claude Code:

.. code-block:: text

   /plugin marketplace add ./tinyml-agent-skills

2. Once you have added the marketplace, install the plugin:

.. code-block:: text

   /plugin install tinyml-agent-skills@tinyml-agent-skills

2. Reload plugins for the installation to start reflecting:

.. code-block:: text

   /reload-plugins

Once installed, invoke it with:

.. code-block:: text

   /tinyml-agentic-skill

Or trigger naturally:

.. code-block:: text

   I want to create an ML model for [your-device]
   Train and deploy to embedded device
   Build a Tiny ML model with tinyml-tensorlab
   Deploy a model to [target-device]
   I want to develop an AI solution for...

Step 3: Confirm Tensorlab Installation
---------------------------------------

When you first invoke the skill, it will ask:

.. code-block:: text

   What is the full path to your tinyml-tensorlab directory?
   Example: /home/user/tinyml-tensorlab

Provide the absolute path (not ``~/tinyml-tensorlab``), and the skill will verify
dependencies and set up required environment variables.

.. note::
   **For Device Deployment** — If you plan to deploy the trained model to a TI MCU,
   it is **strongly recommended** to use Claude Code within Code Composer Studio (TI CCStudio IDE).
   This ensures seamless project creation, automatic file generation, and build integration
   without the need for manual setup.

Workflow Overview
=================

The skill guides you through these steps:

**Phase 1: Project Setup**

1. Confirm tinyml-tensorlab installation path
2. Select task type (classification, anomaly detection, regression, forecasting)
3. Choose target device (F28P55, MSPM0, AM26x, etc.)
4. Provide dataset location and channel count
5. Name your project

**Phase 2: Data Preparation**

6. Validate dataset format (auto-fixes common issues)
7. Analyze statistical properties
8. Recommend feature extraction transforms and data processing presets
9. Review and approve data preparation configuration

**Phase 3: Model Configuration**

10. View ranked model recommendations (sorted by size, speed, device fit)
11. Select quantization mode (0=float32, 1=standard, 2=NPU-optimized) - Automatic Mixed Precision Quantization will be applied by default
12. Choose compilation preset

**Phase 4: Training & Deployment**

13. Review complete configuration (all 50+ parameters)
14. Approve and start training
15. Generate compiled binaries and Code Composer Studio project (optional)
16. Deploy to device via CCS or use trained model on PC

Example Interaction
===================

.. code-block:: text

   You: I have motor vibration data and want to detect bearing faults on F28P55

   Skill: Let me set up a fault detection project.

   Questions asked:
   - Dataset location? → /home/user/motor_data/
   - How many channels? → 3 (X, Y, Z accelerometers)
   - Roughly how many samples? → 5000

   Skill: Analyzing dataset...
   → Found 3 classes, 1500 samples per class ✓
   → Recommended FFT-based feature extraction for vibration
   → Recommended CLS_1K_NPU model
   → Quantization mode 2 recommended (F28P55 has NPU)
   [Automatic Mixed Precision Quantization used by default]

   You: Sounds good, proceed with training.

   Skill: Training in progress...
   Epoch 1/15: Loss 0.52, Accuracy 83.2%
   Epoch 5/15: Loss 0.28, Accuracy 92.7%
   Epoch 15/15: Loss 0.18, Accuracy 95.4%

   Compiled model: 38 KB (FLASH), 12 KB (SRAM)

   Deploy to device?

   You: Deploy to device, this looks good.

   Skill: Creating Code Composer Studio project...
   → Project ready at: ~/projects/motor_fault_detector/
   → Open in CCS, build, and click "Flash"

Key Concepts
============

**Quantization Modes**

* **Mode 0 (Float32)** — No compression. Largest model, slowest. Use only for PC-based testing.
* **Mode 1 (Standard PyTorch Quantization)** — 4× smaller. Works on all devices. Recommended default.
* **Mode 2 (NPU-Optimized)** — Smallest, fastest. Requires NPU hardware (F28P55, specific AM26x models).

**Data Transforms**

* **FFT-based** — For frequency-domain patterns (vibration, audio, motor analysis)
* **Raw Transforms** — For time-domain signals (sensor time-series, raw accelerometer)

**Memory Footprint**

After compilation, check both **FLASH** and **SRAM**. Ensure both fit within your device's memory constraints before deployment.

Common Questions
================

**Q: Do I need a GPU?**

A: No, but strongly recommended. CPU training works but is slower.

**Q: What if I want to experiment with different configurations?**

A: Start a new session anytime. Each project is independent. You can run multiple
experiments in parallel with different quantization modes, models, or transforms.

**Q: Can I export the trained model for use outside tinyml-tensorlab?**

A: The skill generates ONNX and compiled C code. You can use the C code in your own
embedded projects. ONNX can be used with any ONNX-compatible framework.

Getting Help
============

**During the Skill Session**

Ask questions anytime:

.. code-block:: text

   Explain quantization
   What does this feature transform do?
   Why do you recommend this model?
   Show me the complete configuration

The skill pauses and explains before proceeding.

**Reporting Issues**

* Report bugs or suggest improvements via GitHub Issues
* Provide: dataset size, target device, error message, and steps to reproduce

Next Steps
==========

1. **Install tinyml-tensorlab**
2. **Prepare your dataset** (CSV, TXT, or NumPy format)
3. **Invoke the skill** — Use ``/tinyml-agentic-skill`` or natural language
4. **Follow guided prompts** — Answer questions about your task and device
5. **Review recommendations** — Approve data transforms, model selection, quantization
6. **Train your model**
7. **Deploy to device**  — Create CCS project and flash to MCU

For sample datasets and quick testing, start with examples in
``tinyml-tensorlab/tinyml-modelzoo/examples/``.