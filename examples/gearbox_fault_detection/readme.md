# Gearbox Fault Detection

## Overview

The Gearbox Fault Detection application is an Edge AI solution that classifies gearbox operating conditions into two states—healthy operation or broken tooth fault—using vibration sensor data in real-time. This enables predictive maintenance on embedded devices and prevents costly equipment failures in industrial settings.

## Problem and Solution

- Gearbox failures are a leading cause of unplanned downtime in industrial machinery
- Traditional condition monitoring requires manual inspection or complex on-site diagnostics
- Early fault detection prevents catastrophic failures and extends equipment lifespan
- Edge AI enables real-time fault diagnosis directly on resource-constrained microcontrollers without cloud connectivity

## Key Performance Targets

- Real-time classification of gearbox operating conditions
- High accuracy on vibration-based fault detection
- Low memory footprint suitable for MCU deployment

## System Components

**1. Hardware:**

- MSPM0G5187 microcontroller with integrated NPU https://www.ti.com/product/MSPM0G5187

**2. Software:**

- Code Composer Studio 12.x or later
- MSPM0 SDK 2.10.00 or later
- TI Edge AI Studio

## Dataset Information

The example uses the **SpectraQuest Gearbox Fault Diagnostics Simulator** dataset:

- **Source:** [Gearbox Fault Diagnosis - Kaggle](https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis)
- **Classes:** 2 (Healthy, Broken Tooth)
- **Samples per class:** 10 operating conditions (0-90% load)

Each CSV file contains:
- **Columns:** `a1`, `a2`, `a3`, `a4` (4 vibration sensors placed in four different direction)
- **Rows:** ~88,000 samples per file
- **Naming Convention:** `{b|h}30hz{load}.csv`

## Feature Extraction Pipeline

1. Sensor Input: 4-channel accelerometer data
2. Windowing: Extract fixed-length windows (256 samples) from continuous vibration data
3. Single frame input to model

## Model Architecture Options

The example includes multiple optimized 1D CNN-based models targeting different deployment scenarios:

| Model | Parameters | Flash (bytes) | Description |
|-------|-----------|---------------|-------------|
| **GearboxFault_model_1.2k** | ~1,174 | ~8,233 | 4-layer network with progressive channel reduction (12→12→8→8) |
| **GearboxFault_model_1.5k** | ~1,914 | ~8,878 | 3-layer network with constant channels (16→16→16) |

Both models use:
- INT8 quantization for reduced memory footprint
- 4-channel input, 256-sample sequences

## Model Performance

- Accuracy: 97-100% depending on model selection
- Both models achieve >97% accuracy with varying resource trade-offs

## Training and Deployment Process

NOTE: Running the config yaml handles everything including dataset loading, feature extraction, training, quantization, and compilation.

1. **Training:**
   - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
   - Batch size: 32-64
   - Training epochs: 20-30
   - Optimizer: Adam with learning rate 0.001

2. **Quantization:**
   - INT8 quantization for reduced model size
   - Maintains accuracy while enabling MCU deployment

3. **Compilation:**
   - TI Neural Network Compiler converts trained model
   - Generates model artifacts for device deployment

## How to Run

After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

**Windows:**
```bash
.\run_tinyml_modelzoo.bat examples\gearbox_fault_detection\config_MSPM0.yaml
```

**Linux:**
```bash
./run_tinyml_modelzoo.sh examples/gearbox_fault_detection/config_MSPM0.yaml
```

## References

- [SpectraQuest Gearbox Dataset on Kaggle](https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis)
- [TI Neural Network Compiler User Guide](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- [TI Model Training Guide](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main)
- [EdgeAI Software Guide](https://dev.ti.com/tirex/explore/node?node=A__AKCnvqDed-Plz2JO5Umb3Q__MSPM0-SDK__a3PaaoK__LATEST)
