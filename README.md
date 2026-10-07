# Tiny ML Agent - Claude Code Skill

An end-to-end Claude Code plugin for building, training, compiling, and deploying ML models to embedded MCUs using tinyml-modelzoo. Automates the complete workflow from data validation through device deployment.

**Status:** Beta — Workflow and output may change.

## Skills Included

| Skill | Command | Purpose |
|-------|---------|---------|
| **setup** | `/tinyml-agent-skills:setup` | One-time configuration: update mode, tinyml-modelzoo path, virtual environment, package installation (incl. AutoNAS), optional verification training |
| **tinyml-workflow-agent** | `/tinyml-agent-skills:tinyml-workflow-agent` | End-to-end workflow: config creation, training, compilation, CCS project build, and device flashing. Can start at Step 1 or jump to any intermediate step |

## Installation

### Prerequisites

- `tinyml-modelzoo` cloned on your system
- Python 3.10+ (setup creates or detects a virtual environment and installs the TinyML packages)
- **Code Composer Studio (CCS)** with the SDK for your target device (C2000Ware, MSPM0 SDK, or MCU+ SDK). The workflow imports an SDK AI-example template into a CCS project and uses it as the working directory
- A TI compiler (C2000 CGT or ARM CGT) for compilation
- A cleaned and ready-to-use dataset (CSV, TXT, or NumPy format)

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

Setup walks through:
1. **Script discovery** — Locates the plugin's `runner.py` scripts
2. **Update mode** — Pinned or auto-update (see below)
3. **tinyml-modelzoo path** — Prompts for the path and verifies it (`examples/`, `run_tinyml_modelzoo.sh`, `tinyml_modelzoo/`)
4. **Virtual environment & packages** — Uses a `.venv` inside the modelzoo directory if present; otherwise asks whether to use system Python or create a new `.venv`. Verifies `tinyml_modelmaker`, `tinyml_tinyverse`, `tinyml_torchmodelopt`, and `tinyml_modelzoo` import cleanly, installs the AutoNAS MCP (custom dataset formats and Neural Architecture Search) if missing, and records the installed package paths
5. **Verification training (optional)** — Runs a small C2000 example end-to-end to confirm training and compilation work
6. **Save configuration** — Writes `~/.tinyml-agent-skills/.env`

Configuration is stored in `~/.tinyml-agent-skills/.env` in your home directory (same location on Linux, macOS, and Windows). It persists across plugin updates, reinstalls, and Claude Code sessions. It holds `IS_REPO_SETUP`, `TINYML_MODELZOO_PATH`, `UPDATE_MODE`, `UPDATE_PINNED_VERSION`, and the modelmaker/tinyverse/modelopt package paths. Re-run setup if you move or reinstall tinyml-modelzoo, or want to change update mode.

### Update Modes

Choose your preferred update strategy during setup:

#### Pinned Mode

Skill remains on the installed version. No automatic updates.

- Use when: Reproducibility is critical and you want to stick to one version
- Manual updates: Re-run setup skill post update or if you want to switch update modes

#### Auto-update Mode

Skill checks for updates at the start of each session.

- **How it works:** Fetches the tinyml-modelzoo repository and compares the local checkout against `origin/main`
- **When updates occur:** Checked on every session start. If updates are available, you are asked for confirmation before `git pull` runs
- **Use when:** You want latest features and improvements automatically

Your choice persists in `~/.tinyml-agent-skills/.env` and survives session restarts, plugin updates, and reinstalls.

## Workflow Overview

After setup, invoke the tinyml-workflow-agent skill. Every session begins with a **session setup** (load `.env`, activate the venv, check for updates, probe CCS MCP availability), then runs the steps below. Configs, training outputs, and compilation artifacts all live in the imported **CCS project directory** (`$CCS_PROJECT_PATH`), not in tinyml-modelzoo.

| Step | Description |
|------|-------------|
| 1 | **Session setup** — load `.env`, activate venv, check for updates, detect CCS MCP servers |
| 2 | **Requirements** — task type and target device (valid values are read from modelmaker source) |
| 3 | **Early CCS import** — find the SDK, locate the AI example template for your device, import it into your CCS workspace |
| 3.5 | **Example check** — if a tinyml-modelzoo example matches your task and device, choose: use it as-is (bundled dataset + config), use its config as a template with your own dataset, or build from scratch |
| 4 | **Dataset & task name** — dataset location and channel count |
| 5 | **Common section** — generate the `common` config section |
| 6 | **Dataset validation** — validate format and auto-fix common issues |
| 7 | **Dataset section** — generate the `dataset` config section |
| 8A | **Dataset analysis** — statistical insights used to guide later choices |
| 8B | **Feature extraction & data processing** — recommended presets/transforms (FFT-based, raw, multi-frame) |
| 8 | **Model selection** — ranked recommendations from tinyml-modelzoo |
| 9 | **Training section** — model, hyperparameters, quantization mode, NAS, GPU usage |
| 10 | **Testing section** |
| 11 | **Compilation section** — device-aware preset recommendations |
| 12 | **Assemble `config.yaml`** — written to `$CCS_PROJECT_PATH` with absolute output paths |
| 13 | **Run training & compilation** — compiler path is discovered (via CCS MCP or filesystem) and exported in the same shell call; logs stream live |
| 13A | **Verify artifacts** — `mod.a` present and FLASH/SRAM footprint shown and acknowledged (blocking gate) |
| 14 | **Deploy** — exclude `training/`, `compilation/`, `config.yaml` from the CCS build, copy artifacts and golden vectors, build, flash |

After every step that generates a config section, the skill shows you the config built so far and proceeds only with your approval.

**Mid-workflow entry:** If you already have a config, trained model, or compiled artifacts, you can jump straight to the relevant step (e.g. "just run training on my config" → Step 13; "just deploy my compiled model" → Step 14). Session setup still runs first.

## Key Features

- **End-to-end automation** — Handles configuration, training, compilation, and deployment in one workflow
- **CCS-integrated** — Works inside a CCS project imported from the SDK template; builds and flashes through the CCS MCP servers (`ccs-project`, `ccs-debug`) when available, falling back to headless dslite otherwise
- **Example reuse** — Matches tinyml-modelzoo examples to your task/device and lets you reuse their config and dataset
- **Device-aware optimization** — Recommends quantization modes and presets specific to your MCU (F28P55, MSPM0, AM26x, etc.) and NPU availability
- **Intelligent data handling** — Validates dataset format, detects common issues, applies fixes automatically
- **AutoNAS** — Custom dataset formats and Neural Architecture Search, installed during setup
- **Guided decisions** — Explains trade-offs for advanced features including quantization, and feature extraction presets/transforms
- **Error diagnosis** — Captures failures, suggests fixes, and re-runs until training completes cleanly
- **Memory verification** — Tracks FLASH/SRAM usage and ensures the model fits the target device before deployment

## Data Requirements

The skill handles format conversion for tinyml-modelzoo compatibility. You must provide:
- **Pre-processed data** — Cleaned, normalized, and formatted (CSV, TXT, or NumPy). The skill does not perform raw data cleaning or preprocessing
- **Appropriate size** — Dataset must be suitable for your task type and target device constraints

## Getting Started

### First Time: Complete Setup

Run the setup skill immediately after installing the plugin. This is a one-time configuration:

```
/tinyml-agent-skills:setup
```

### Trigger the Workflow

```
/tinyml-agent-skills:tinyml-workflow-agent
```

Or use natural language:
- "Create an ML model for my MCU"
- "Train and deploy to embedded device"
- "Deploy a model to F28P55"
- "Build a Tiny ML model with tinyml-modelzoo"

### Workflow Execution

**Phase 1: Project Definition & CCS Import**
- Select task type (classification, anomaly detection, regression, forecasting) and target device
- Import the SDK AI-example template into a CCS project
- Optionally reuse a matching tinyml-modelzoo example (config and/or dataset)

**Phase 2: Data Analysis & Preparation**
- Validate dataset format (auto-corrects common issues)
- Review statistical properties and data distribution
- Select feature extraction transforms and data processing presets

**Phase 3: Model & Training Configuration**
- Review ranked model recommendations from tinyml-modelzoo
- Select quantization mode:
  - `0` = Float32 (no compression, largest model, slowest)
  - `1` = Standard quantization (4× smaller, all devices)
  - `2` = NPU-optimized (smallest model, fastest, requires NPU hardware)
- Configure hyperparameters and training settings (GPU use is offered if detected)

**Phase 4: Build, Compile & Deploy**
- Generate the complete `config.yaml` in the CCS project and review it
- Train and compile; check artifacts and the FLASH/SRAM footprint
- Build the CCS project with golden vectors and flash the device

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