================================================
Tiny ML Agent Skill — AI-Guided Model Creation
================================================

.. warning::
   **Beta Feature** — This skill is experimental. Workflow and output may change.

What Is This Skill?
====================

The **Tiny ML Agent Skill** is an end-to-end AI assistant that guides you through
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
* **Your Dataset** — CSV, TXT, or NumPy format with pre-processed, cleaned sensor or image data

  * Sample datasets are included in ``tinyml-tensorlab/tinyml-modelzoo/examples/`` for testing

**Optional:**

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

   /plugin marketplace add path/to/tinyml-agent-skills

**NOTE:** If you have cloned tinyml-tensorlab, tinyml-agent-skills can be found at ``tinyml-tensorlab/tinyml-agent-skills``

2. Once you have added the marketplace, install the plugin:

.. code-block:: text

   /plugin install tinyml-agent-skills@tinyml-agent-skills

3. Reload plugins for the installation to start reflecting:

.. code-block:: text

   /reload-plugins

Step 3: Run Setup Skill (First Time Only)
------------------------------------------

Before using the workflow skill, run the setup skill [needed for initial setup **ONLY**]:

.. code-block:: text

   /tinyml-agent-skills:setup

This configures update mode, discovers script directories, verifies tinyml-tensorlab installation, and saves all required variables to ``~/.tinyml-agent-skills/.env``.

Step 4: Invoke the Workflow Skill
----------------------------------

Once setup is complete, invoke the main workflow skill with:

.. code-block:: text

   /tinyml-agent-skills:tinyml-workflow-agent

Or trigger naturally:

.. code-block:: text

   I want to create an ML model for [your-device]
   Train and deploy to embedded device
   Build a Tiny ML model with tinyml-tensorlab
   Deploy a model to [target-device]
   I want to develop an AI solution for...

Setup Confirmation
-------------------

When you run the setup skill, it will prompt:

.. code-block:: text

   What is the full path to your tinyml-tensorlab directory?
   Example: /home/user/tinyml-tensorlab

   Update mode?
   → auto-update (latest skill version on each run)
   → pinned (lock to current version, manual updates only)

.. note::
   Updates cover **both** the tinyml-agent-skill and tinyml-tensorlab. When you enable auto-update mode, both components are kept in sync with the main repository branch.

Provide the absolute path (not ``~/tinyml-tensorlab``) and select the update mode of your choice. The setup skill will:

* Configure update mode (auto-update or pinned to specific version)
* Verify tinyml-tensorlab installation and dependencies
* Discover SCRIPTS_DIR location
* Set up required environment variables and save to ``~/.tinyml-agent-skills/.env``

.. note::
   **For Device Deployment** — If you plan to deploy the trained model to a TI MCU,
   it is **strongly recommended** to use Claude Code within Code Composer Studio (TI CCStudio IDE).
   This ensures seamless project creation, automatic file generation, and build integration
   without the need for manual setup.

Workflow Overview
=================

The workflow skill guides you through 13 steps across 4 phases:

**Phase 1: Project Configuration (Steps 1-3)**

1. Specify task type (classification, anomaly detection, regression, forecasting)
2. Choose target device (F28P55, MSPM0, AM26x, etc.)
3. Provide dataset location and variable (indepenent features) count

**Phase 2: Data Preparation (Steps 4-6)**

4. Validate dataset format (auto-fixes common issues)
5. Analyze statistical properties
6. Select feature extraction transforms and data processing presets (FFT-based or Raw transforms)

**Phase 3: Model & Training Configuration (Steps 7-10)**

7. View ranked model recommendations (sorted by size, speed, device fit)
8. Select quantization mode:

   * Mode 0 (Float32) — No compression, largest model, PC testing only
   * Mode 1 (Standard) — 4× smaller, works on all devices
   * Mode 2 (NPU-Optimized) — Smallest/fastest, requires NPU hardware (F28P55, specific AM26x models)
   * Automatic Mixed Precision Quantization applied by default

9. Choose compilation preset
10. Generate ``config.yaml`` with all 50+ parameters

**Phase 4: Training & Deployment (Steps 11-13)**

11. Upon user review & approval of configuration, start training; show metrics and compiled model size i.e memory footprint (FLASH/SRAM)
12. Create and build Code Composer Studio project
13. Deploy to device via CCS or use trained model on PC

Key Concepts
============

**Quantization Modes**

* **Mode 0 (Float32)** — No compression. Largest model, slowest. Use only for PC-based verification.
* **Mode 1 (Standard PyTorch Quantization)** — 4× smaller. Works on all devices. Recommended default.
* **Mode 2 (NPU-Optimized)** — Smallest, fastest. Requires NPU hardware (F28P55, specific AM26x models).

**Feature Extraction**

* **FFT-based** — For frequency-domain patterns (vibration, audio, motor analysis)
* **Raw Transforms** — For time-domain signals (sensor time-series, raw accelerometer)
* **Multi-frame** — Captures temporal context across multiple samples

**Memory Footprint**

After compilation, check both **FLASH** and **SRAM**:

* **FLASH** — Model weights and code (read-only memory)
* **SRAM** — Runtime working memory (read-write memory)

Ensure both fit within your device's memory constraints before deployment.

Common Questions
================

**Q: Do I need a GPU?**

A: No, but strongly recommended. CPU training works but is slower.

**Q: What if I want to experiment with different configurations?**

A: Start a new session anytime. Each project is independent. You can run multiple
experiments in parallel with different quantization modes, models, or transforms.

**Q: Can I export the trained model for use outside tinyml-tensorlab?**

A: The skill generates ONNX and compiled model artifacts. You can use the artifacts in your own
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

Next Steps
==========

1. **Install tinyml-tensorlab**
2. **Prepare your dataset** (CSV, TXT, or NumPy format)
3. **Invoke the skill** — Use ``/tinyml-agent-skills:tinyml-workflow-agent`` or natural language
4. **Follow guided prompts** — Answer questions about your task and device
5. **Review recommendations** — Approve data transforms, model selection, quantization
6. **Train your model**
7. **Deploy to device**  — Create CCS project and flash to MCU

For sample datasets and quick testing, start with examples in
``tinyml-tensorlab/tinyml-modelzoo/examples/``.