# Tiny ML Agent - Claude Code Skill

An end-to-end Claude Code plugin for building, training, compiling, and deploying ML models to embedded MCUs using tinyml-tensorlab. Automates the complete workflow from data validation through device deployment.

**Status:** Beta — Workflow and output may change.

## Installation

### Prerequisites

- `tinyml-tensorlab` installed and configured on your system
- Python environment with PyTorch and tinyml-tensorlab dependencies
- A cleaned and ready-to-use dataset (CSV, TXT, or NumPy format)

**Note:** `tinyml-tensorlab` already includes `tinyml-agent-skills`. If the directory exists locally, skip the clone step.

### Register Plugin Marketplace

Add this repository as a marketplace in Claude Code:
```
/plugin marketplace add path/to/tinyml-agent-skills 
# Replace path/to/tinyml-agent-skills with your actual path to the directory
```

### Install the Plugin

```
/plugin install tinyml-agent-skills@tinyml-agent-skills
```

## Initial Setup (Required)

**Before using the tinyml-workflow-agent skill, complete the one-time setup:**

```
/tinyml-agent-skills:setup
```

This configures:
- **tinyml-tensorlab path** — Locates and verifies your installation
- **Update mode** — Selects how skill updates are managed
- **Environment variables** — Saves configuration to `~/.tinyml-agent-skills/.env`

Configuration is stored in `~/.tinyml-agent-skills/.env` in your home directory. This file persists across plugin updates, reinstalls, and Claude Code sessions. Re-run setup only if you move tinyml-tensorlab or want to change update mode.

### Update Modes

Choose your preferred update strategy during setup:

#### Pinned Mode

Skill remains on the installed version. No automatic updates.

- Use when: Reproducibility is critical and you want to stick to one version
- Manual updates: Re-run setup skill post update or if you want to switch update modes

#### Auto-update Mode

Skill automatically checks for and applies updates from the main branch of the tinyml-tensorlab repository.

- **How it works:** Compares the installed version against the main branch of the tinyml-tensorlab repository
- **When updates occur:** Updates are checked for during startup of the skill each time. If available, detected updates are auto-installed post user confirmation
- **Use when:** You want latest features and improvements automatically

Your choice persists in `~/.tinyml-agent-skills/.env` and survives session restarts, plugin updates, and reinstalls.

## Workflow Overview

Post the setup, you may invoke the tinyml-workflow-agent skill which guides you through the complete Tiny ML Tensorlab pipeline:

1. **Project Configuration** — Define task type (classification, anomaly detection, regression, forecasting) and select target device
2. **Data Analysis** — Validate dataset format, analyze statistical properties, auto-fix common issues
3. **Feature Extraction** — Select and configure feature transforms (FFT-based, raw transforms, multi-frame) based on data characteristics
4. **Model Selection** — Review ranked recommendations from tinyml-modelzoo based on dataset size and device constraints
5. **Training Configuration** — Set hyperparameters, quantization mode, and Neural Architecture Search (NAS) options
6. **Build & Compile** — Generate optimized binaries for your target MCU
7. **Device Deployment** — Create Code Composer Studio project with compiled artifacts and golden vectors for flashing

## Key Features

- **End-to-end automation** — Handles configuration, training, compilation, and deployment in one workflow
- **Device-aware optimization** — Recommends quantization modes and presets specific to your MCU (F28P55, MSPM0, AM26x, etc.) and NPU availability
- **Intelligent data handling** — Validates dataset format, detects common issues, applies fixes automatically
- **Guided decisions** — Explains trade-offs for advanced features including quantization, and feature extraction presets/transforms
- **Error diagnosis** — Captures failures, suggests fixes, validates corrections before proceeding
- **Memory verification** — Tracks FLASH/SRAM usage, ensures model fits on target device

## Data Requirements

The skill handles format conversion for tinyml-tensorlab compatibility. You must provide:
- **Pre-processed data** — Cleaned, normalized, and formatted (CSV, TXT, or NumPy). The skill does not perform raw data cleaning or preprocessing
- **Appropriate size** — Dataset must be suitable for your task type and target device constraints

## Getting Started

### First Time: Complete Setup

As explained above, run the setup skill immediately after installing the plugin. This is a one-time configuration:

```
/tinyml-agent-skills:setup
```

Setup will:
- Verify tinyml-tensorlab installation
- Configure update mode (pinned or auto-update)
- Save configuration to `~/.tinyml-agent-skills/.env`

### Trigger the Skill

Start the tinyml-workflow-agent skill by saying:

```
/tinyml-agent-skills:tinyml-workflow-agent
```

Or use natural language:
- "Create an ML model for my MCU"
- "Train and deploy to embedded device"
- "Deploy a model to F28P55"
- "Build a Tiny ML model with tinyml-tensorlab"

### Workflow Execution

The skill guides you through each phase with clear prompts:

**Phase 1: Project Definition**
- Select task type (classification, anomaly detection, regression, forecasting)
- Choose target device (F28P55, MSPM0, AM26x, etc.)
- Specify dataset path and channel count
- Generate initial project configuration

**Phase 2: Data Analysis & Preparation**
- Validate dataset format (auto-corrects common issues)
- Review statistical properties and data distribution
- Select appropriate feature extraction transforms
- Apply data processing presets based on your data type

**Phase 3: Model & Training Configuration**
- Review ranked model recommendations from tinyml-modelzoo
- Select quantization mode:
  - `0` = Float32 (no compression, largest model, slowest)
  - `1` = Standard quantization (4× smaller, all devices)
  - `2` = NPU-optimized (smallest model, fastest, requires NPU hardware)
- Enable Neural Architecture Search (NAS) for automatic model tuning
- Configure hyperparameters and training settings

**Phase 4: Build, Compile & Deploy**
- Generate complete `config.yaml` with all settings
- Review configuration before proceeding
- Train the model and monitor metrics
- Compile to target MCU binary
- Create Code Composer Studio project with golden vectors
- Deploy to device

**Note:** For best results deploying to CCStudio, use Claude Code within the TI Code Composer Studio IDE to ensure seamless project creation and build integration.

## Technical Concepts

### Quantization Modes

**Mode 0: Float32 (Full Precision)**
- No compression applied
- Largest model size
- Slowest inference
- Use for: Verification and baseline accuracy testing only

**Mode 1: Standard Quantization**
- PyTorch INT8 quantization
- ~4× model size reduction
- Works on all TI MCU devices
- Recommended for: Most production deployments with memory constraints

**Mode 2: NPU-Optimized**
- TI Neural Processing Unit acceleration
- Smallest model footprint
- Fastest inference
- Requires: Target device with NPU hardware (e.g., AM26x series)

### Feature Extraction Strategies

The skill recommends extraction methods based on data characteristics:

**FFT-Based Transforms**
- Use for: Frequency-domain patterns (vibration, acoustic, audio signals)
- Extracts: Power spectrum, frequency bins, harmonic content
- Ideal for: Anomaly detection on vibration or sound data

**Raw Signal Transforms**
- Use for: Time-domain sensor signals (accelerometer, temperature, pressure)
- Extracts: Statistical features (mean, std dev, peak, energy)
- Ideal for: Time-series classification and regression

**Multi-Frame Aggregation**
- Use for: Temporal pattern recognition
- Captures: Context across multiple consecutive samples
- Ideal for: Gesture recognition, activity detection, sequential patterns

### Memory Footprint Management

After compilation, the skill reports:

- **FLASH** — Permanent storage for model weights and inference code (read-only memory)
- **SRAM** — Runtime working memory for activations and intermediate computations (read-write memory)

The skill automatically verifies that both metrics fit within your target device's constraints before deployment proceeds.