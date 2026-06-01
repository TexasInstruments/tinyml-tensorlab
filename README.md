# TinyML Agent - Claude Code Skill

This repository contains the tinyml-agentic-skill, an end-to-end skill for building, training, compiling, and deploying ML models to embedded MCUs using tinyml-tensorlab.

## Installation

### Add to Claude Code Marketplace

You can register this repository as a Claude Code Plugin marketplace by running the following command in Claude Code:
```
/plugin marketplace add ssh://git@bitbucket.itg.ti.com/tinyml-algo/tinyml-agent-skills.git
```

### Install the Plugin

Once you have added the marketplace, install the plugin:
```
/plugin install tinyml-agent@tinyml-agentic-skills
```

## What This Skill Does

The TinyML Agent guides you through the complete TinyML workflow to:

1. **Configure your project** — Select task type, target device, and data sources
2. **Analyze your data** — Extract statistical insights and validate dataset format
3. **Extract features** — Generate baseline feature extraction and data processing transforms
4. **Select a model** — Ranked model recommendations from tinyml-modelzoo based on dataset size and input variable constraints
5. **Configure training** — Set hyperparameters, quantization mode, and Neural Architecture Search (NAS) options
6. **Build and compile** — Generate compiled binaries for your target MCU
7. **Deploy to hardware** — Create, and build a new Code Composer Studio project with the compiled binaries and golden vectors for device deployment.

## Why Use This Skill

- **Complete workflow automation** — Handles all configuration details from raw data to deployed model
- **Device-aware optimization** — Recommends presets and quantization modes specific to your target MCU (F28P55, MSPM0, AM26x, etc.) depending on NPU availability
- **Data validation** — Automatically detects and fixes common dataset formatting issues
- **Guided decision-making** — Explains trade-offs for quantization, NAS, and feature extraction strategies
- **Error recovery** — Captures and diagnoses failures, suggests fixes, and validates fixes before proceeding
- **Memory footprint tracking** — Displays FLASH/SRAM usage and verifies model fits on device before deployment

**Prerequisites:**
- `tinyml-tensorlab` installed and set up on your system
- A dataset in CSV, TXT, or NumPy format
- Python environment with PyTorch and tinyml dependencies

## How to Use

### Quick Start

Invoke the skill in Claude Code by mentioning any of:
- "Create an ML model for my MCU"
- "Train and deploy to embedded device"
- "Build a TinyML model with tinyml-tensorlab"
- "I want to deploy a model to [your-target-device]"
- "I want to develop an AI solution for..."

[Above list is not exhaustive, and is only for representational purposes]

Or use the explicit skill trigger:
```
/tinyml-agentic-skill
```

### Step-by-Step Workflow

#### **Setup (Before Step 1)**
Confirm your `tinyml-tensorlab` installation path. The skill will set up required environment variables and verify dependencies.

#### **Steps 1-3: Project Configuration**
- Specify task type (classification, anomaly detection, etc.)
- Choose target device (F28P55, MSPM0, AM26x, etc.)
- Provide dataset location and channel count
- Generate project configuration

#### **Steps 4-6: Data Preparation**
- Validate dataset format (auto-fixes common issues)
- Analyze statistical properties
- Select feature extraction transforms and data processing presets
- Review recommendations before proceeding

#### **Steps 7-10: Model & Training Config**
- View ranked model recommendations
- Select quantization mode (0=float32, 1=standard, 2=NPU-optimized)
- Enable Neural Architecture Search (NAS) if desired
- Choose compilation preset

#### **Step 11: Generate Config**
- Assemble all sections into `config.yaml`
- Review complete configuration
- Approve before training

#### **Steps 12-13: Train & Deploy**
- Start training
- View training metrics and compiled model size
- Create and build Code Composer Studio project
- Ready for flashing to MCU and on-device use.

### Configuration Example

During setup, the skill guides you through decisions like:

```
Task Type?
  → classification, anomaly_detection, regression, forecasting

Target Device?
  → F28P55, MSPM0, AM26x, etc.

Quantization Mode?
  → 0 (float32, no compression)
  → 1 (standard quantization, all devices)
  → 2 (NPU-optimized, for TI NPU acceleration)
```

## Key Concepts

### Quantization Modes
- **Mode 0 (Float32)** — No compression. Largest model, slowest inference. Use only for verification.
- **Mode 1 (Standard)** — PyTorch quantization. 4× smaller, works on all devices.
- **Mode 2 (NPU-Optimized)** — TI NPU acceleration. Smallest model, fastest inference. Requires NPU hardware.

### Feature Extraction
The skill recommends presets based on your data:
- **FFT-based** — For frequency-domain patterns (e.g., vibration, audio)
- **Raw transforms** — For time-domain signals (e.g., sensor time-series)
- **Multi-frame** — Captures temporal context across multiple samples

### Memory Footprint
After compilation, the skill displays:
- **FLASH** — Model weights and code (read-only memory)
- **SRAM** — Runtime working memory (read-write memory)

Verify these fit within your device's capabilities before deployment.

### Recovery & Troubleshooting

If the skill reports errors:
- Read the error message carefully (e.g., "frame_size > min_sequence_length")
- Follow the suggested fix (e.g., "reduce frame_size to ≤70")
- Restart at the current step with corrected parameters