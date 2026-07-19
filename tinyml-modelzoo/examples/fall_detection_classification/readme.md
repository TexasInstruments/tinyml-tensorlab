# Human Fall Detection Classification

## Overview

The Human Fall Detection Classification application is an Edge AI solution that classifies human movement into two states: Activities of Daily Living (ADL) or Fall, using accelerometer data in real-time. This enables safety monitoring on embedded devices and provides immediate alerts in belt-mounted safety systems for vulnerable individuals such as elderly users or industrial workers.

## Problem and Solution

- Falls are a leading cause of injury and fatality among elderly individuals and in industrial environments
- Traditional fall detection systems rely on cloud connectivity or bulky wearable hardware
- Early and accurate fall detection enables timely emergency response and reduces injury severity
- Edge AI enables real-time fall classification directly on resource-constrained microcontrollers without cloud connectivity

## Key Performance Targets

- Real-time classification of human movement into ADL or Fall
- High accuracy on accelerometer-based fall detection (≥97.5%)
- Low memory footprint suitable for MCU deployment

## System Components

**1. Hardware:**

- MSPM0G5187 microcontroller with integrated NPU https://www.ti.com/product/MSPM0G5187
- TIDA-010997 EdgeAI Boosterpack with BMI270 accelerometer

**2. Software:**

- Code Composer Studio 12.x or later
- MSPM0 SDK 2.11.00 or later
- TI Edge AI Studio

## Dataset Information

The example uses the **SisFall** dataset as the primary training dataset:

- **Source:** [SisFall: A Fall and Normal Movement Dataset](https://www.mdpi.com/1424-8220/17/1/198)
- **Classes:** 2 (ADL, Fall)
- **Total Files:** ~4500

Each file contains readings from three sensors:
- **ADXL345** — 13-bit triaxial accelerometer
- **ITG3200** — triaxial gyroscope
- **MMA8451Q** — triaxial accelerometer

### Data Cleaning and Preprocessing

The SisFall dataset contains data from three sensors, but only the **ADXL345 accelerometer** data is used for this example to match the single-accelerometer hardware setup (BMI270 on TIDA-010997). The following cleaning and preprocessing steps are applied:

**1. Sensor Filtering:**
- Only ADXL345 (13-bit triaxial accelerometer) columns are retained
- ITG3200 (gyroscope) and MMA8451Q (accelerometer) columns are discarded

**2. Bit-Depth Scaling:**
- The ADXL345 is a 13-bit accelerometer, while the target hardware uses the BMI270 which is a 16-bit accelerometer
- To match the input resolution expected by the firmware, all ADXL345 readings are scaled up from 13-bit to 16-bit resolution using the formula: `scaled_value = raw_value * (2^16 / 2^13) = raw_value * 8`
- This ensures consistency between the training data distribution and the live sensor data fed during inference on the MCU

**3. Format Conversion:**
- The cleaned and scaled dataset is converted into a format compatible with tinyml-tensorlab for model training
- Each file is parsed, labeled (ADL or Fall), and exported into the required input structure

## Feature Extraction Pipeline

1. Sensor Input: 3-axis accelerometer data (x, y, z) from ADXL345 (scaled to 16-bit)
2. Real FFT: 256-point FFT using ARM CMSIS-DSP
3. Complex Magnitude Calculation
4. DC Removal
5. Binning: Average 16 adjacent FFT bins → 8 features
6. Frame Concatenation: Stack 8 frames (64 total features per axis)

## Model Architecture

A generic time-series classification CNN model is used:

| Model | Parameters | Flash (kB) | RAM (kB) | Inference Latency (NPU) | Accuracy |
|-------|-----------|------------|----------|--------------------------|----------|
| **CLS_6k** | ~6,000 | 14 | 0.7 | 0.67 ms | 97.65% |

_NOTE: The above statistics was measured on LP-MSPM0G5187 Launchpad_

The model uses:
- INT8 quantization for reduced memory footprint
- 3-channel input (x, y, z axes)

## Training and Deployment Process

NOTE: Running the config yaml handles everything including dataset loading, data cleaning, feature extraction, training, quantization, and compilation.

1. **Training:**
   - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
   - Batch size: 256
   - Training epochs: 50

2. **Quantization:**
   - INT8 quantization for reduced model size
   - Maintains accuracy while enabling MCU deployment

3. **Compilation:**
   - TI Neural Network Compiler converts the trained model
   - Generates model artifacts for device deployment

## How to Run

After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

**Windows:**
```bash
.\run_tinyml_modelzoo.bat examples\fall_detection_classification\config_MSPM0.yaml
```

**Linux:**
```bash
./run_tinyml_modelzoo.sh examples/fall_detection_classification/config_MSPM0.yaml
```

## References

- [SisFall Dataset](https://www.mdpi.com/1424-8220/17/1/198)
- [TI Neural Network Compiler User Guide](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/)
- [TI Model Training Guide](https://github.com/TexasInstruments/tinyml-tensorlab/tree/main)
- [EdgeAI Software Guide](https://dev.ti.com/tirex/explore/node?node=A__AKCnvqDed-Plz2JO5Umb3Q__MSPM0-SDK__a3PaaoK__LATEST)
