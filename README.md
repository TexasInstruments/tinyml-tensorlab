# Tiny ML Agent - Claude Code Skill

An end-to-end Claude Code plugin for building, training, compiling, and deploying ML models to embedded MCUs using tinyml-modelzoo. Automates the complete workflow from data validation through device deployment.

**Status:** Beta — Workflow and output may change.

## Skills Included

| Skill | Command | Purpose |
|-------|---------|---------|
| **setup** | `/tinyml-agent-skills:setup` | One-time configuration: tinyml-modelzoo path, virtual environment, package installation, optional verification training [Currently Not Used in the officially released version (i.e. in `ti-ccstudio-tinyml-workflow` skill)] |
| **ti-ccstudio-tinyml-workflow** | `/tinyml-agent-skills:ti-ccstudio-tinyml-workflow` | End-to-end workflow: config creation, training, compilation, CCS project build, and device flashing. Can start at Step 1 or jump to any intermediate step |

## Installation

### Prerequisites

- CCStudio IDE or VS Code IDE with CCStudio Extension installed
- TI CCStudio Edge AI Toolkit

**Currently we make the tinyml-workflow skill available via only the ti-ccstudio suite of AI Skills & Agents. Future releases aim to make a generic version available that is accessible via the terminal as well.**

All initial setup tasks including skill installation into the workspace's .claude/ directory, .env file generation, and its subsequent population are taken care of by the TI CCStudio Edge AI Toolkit (and CCStudio Extension in VS Code IDE) automatically.

## Workflow Overview

The ti-ccstudio-tinyml-workflow skill automates the complete ML model lifecycle on embedded devices. Every invocation begins with a **session setup** phase (load configuration, verify environment, detect CCS MCP servers), then proceeds through configurable workflow steps. All configs, training outputs, and compilation artifacts live in the **CCS project directory** (`$CCS_PROJECT_PATH`), not in tinyml-modelzoo.

### Workflow Phases

| Phase | Steps | Purpose |
|-------|-------|---------|
| **Project Definition & CCS Import** | 1–3 | Select task type and target device; import SDK AI-example template into CCS workspace |
| **Data Analysis & Preparation** | 4–8 | Validate dataset format; analyze statistics; select feature extraction transforms |
| **Model & Training Configuration** | 9–11 | Choose model, quantization mode, hyperparameters, and compilation settings |
| **Build, Compile & Deploy** | 12–14 | Assemble config, train/compile with live logs, verify artifacts, build and flash device |

### Detailed Step Breakdown

| Step | Description |
|------|-------------|
| **Session Setup** | Load `.env`, verify venv, detect CCS MCP servers/CCS Project CLI tools, probe compiler availability |
| **1. Requirements** | Select task type (classification, anomaly detection, regression, forecasting) and target device; valid values read from modelmaker source |
| **2. CCS Import** | Locate SDK for target device, find AI-example template, import into CCS workspace |
| **3. Example Check** | If a bundled example (`references/examples/`) matches your task and device, optionally reuse its config and/or dataset |
| **4. Dataset & Task Name** | Specify dataset location (CSV, TXT, or NumPy format) and channel count |
| **5. Dataset Validation** | Auto-detect validate compatibility with tinyml-modelzoo |
| **6. Dataset Analysis** | Generate statistical insights (distribution, normalization, imbalance) to guide feature extraction and model selection |
| **7. Feature Extraction & Data Processing** | Review recommended presets/transforms (FFT-based, raw, multi-frame); apply based on data characteristics |
| **8. Model Selection** | Explore ranked recommendations from tinyml-modelzoo; view latency/accuracy/footprint trade-offs |
| **9. Training Section** | Configure model, hyperparameters, quantization mode, NAS settings, and GPU usage |
| **10. Testing Section** | Set test parameters and metrics |
| **11. Compilation Section** | Review device-aware preset recommendations; configure output settings |
| **12. Assemble config.yaml** | Generate and review complete configuration with absolute output paths; written to `$CCS_PROJECT_PATH` |
| **13. Run Training & Compilation** | Execute training and compilation; compiler path auto-discovered (via CCS MCP or filesystem); logs stream live |
| **13A. Verify Artifacts** | Confirm `mod.a` present; display FLASH/SRAM footprint; must fit target device (blocking gate) |
| **14. Deploy** | Exclude training/compilation artifacts and config from CCS build; copy model and golden vectors; build and flash device |

After every step that generates a config section, the skill displays the config built so far and proceeds only with your approval.

**Mid-workflow entry:** Jump to any step if you already have partial progress:
- Start at **Step 9** (training only) if you have a config and dataset
- Start at **Step 13** (compile & verify) if you have a trained model
- Start at **Step 14** (deploy only) if you have compiled artifacts
- Session setup always runs first to verify environment health

## Key Features

- **End-to-end automation** — Single skillCurrently we make the tinyml-workflow skill available via only the ti-ccstudio suite of AI Skills & Agents. Future releases aim to make a generic version available that is accessible via the terminal as well. handles configuration, training, compilation, and device deployment
- **CCS-integrated** — Imports SDK AI-example templates directly into CCS workspace; uses CCS MCP servers (`ccs-project`, `ccs-debug`) for builds and flashing when available, with headless fallback via dslite
- **Flexible entry points** — Start from Step 1 or jump to any intermediate step (e.g., train an existing config, deploy compiled artifacts)
- **Example-driven development** — Bundled examples (`references/examples/`) for each device/task combo; optionally reuse config, dataset, or both
- **Device-aware optimization** — Quantization modes (Float32, INT8, NPU) and compilation presets tuned to specific MCUs (F28P55, MSPM0, AM26x) and hardware capabilities
- **Intelligent data handling** — Auto-detects and fixes common dataset format issues; validates compatibility before training
- **Statistical guidance** — Dataset analysis informs feature extraction strategy and model recommendations
- **Ranked model selection** — Prioritized recommendations from tinyml-modelzoo with latency/accuracy/footprint trade-offs
- **Interactive configuration** — Approval gates after each config section; review and edit before proceeding
- **Live training logs** — Compiler path auto-discovery (CCS MCP or filesystem); training and compilation logs stream in real-time
- **Memory verification** — FLASH/SRAM footprint checked and confirmed before deployment (blocking gate ensures model fits device)
- **Deployment automation** — Artifact staging, golden-vector inclusion, CCS project build, and device flashing in one step

## Data Requirements

The skill auto-detects and validates dataset formats for tinyml-modelzoo compatibility. You must provide:

- **Pre-processed data** — Cleaned, normalized, and formatted in CSV, TXT, or NumPy format. The skill does not perform raw data cleaning; bring data ready for training
- **Appropriate size** — Dataset must be suitable for your task type (classification, anomaly detection, regression, forecasting) and target device memory constraints
- **Supported formats** — CSV, TXT (space/comma-delimited), or NumPy arrays; the skill auto-converts and validates before training

## Getting Started

Invoke the ti-ccstudio-tinyml-workflow skill to begin:

```
/tinyml-agent-skills:ti-ccstudio-tinyml-workflow
```

Or use natural language:
- "Create an ML model for my MCU"
- "Train and deploy to embedded device"
- "Deploy a model to F28P55"
- "Build a Tiny ML model with tinyml-modelzoo"

The skill will guide you through project definition, data preparation, model training configuration, and device deployment. You can start from the beginning or jump to any step if you have existing configs or artifacts.

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