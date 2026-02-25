# PIR Motion Detection

## Overview

  The PIR Motion Detection application is an Edge AI solution that runs on the MSPM0G5187 microcontroller with integrated Neural Processing Unit (NPU). It classifies passive infrared (PIR) sensor signals into different motion categories in real-time, enabling intelligent motion detection for security systems, smart home automation, and occupancy sensing. 

**⚠️ Device Support:** While this documentation focuses on the **MSPM0G5187**, the following device are also fully supported:
  - **CC35X1** 
  - **CC1352**
  - **CC1354** 
  - **CC2755**

Check the `config_<device>.yaml` files for device-specific configurations.

## Problem and Solution

  - Traditional PIR sensors only detect presence/absence without distinguishing motion sources
  - False alarms from pets and environmental factors reduce security system effectiveness
  - Manual threshold tuning is unreliable across different environments and conditions
  - Edge AI enables intelligent motion classification, reducing false positives while maintaining detection sensitivity

## Key Performance Targets

  - Real-time motion classification
  - Greater than 90% classification accuracy
  - Low power consumption for battery-operated devices

## System Components

  1. Hardware:
     - MSPM0G5187 microcontroller with integrated NPU [Link](https://www.ti.com/product/MSPM0G5187)
     - EdgeAI Sensor Boosterpack (TIDA-010997) with PIR sensor

  2. Software:
     - Code Composer Studio 12.x or later
     - MSPM0 SDK 2.08.00 or later
     - TI Edge AI Studio

## Dataset Information

  The example uses the `pir_detection_classification` [Link]( https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/pir_detection_classification_dsk.zip) dataset which contains labeled PIR sensor recordings categorized into three motion classes:

  - **Human Motion**: Movement patterns characteristic of human activity
  - **Background Motion**: Environmental disturbances and noise
  - **Dog Motion**: Movement patterns characteristic of pet dog activity

Data was captured using the EdgeAI Sensor Boosterpack with motion recorded from 6-8 meter range.

## Feature Extraction Pipeline

  1. ADC Sampling
  2. DC Offset Removal
  3. Windowed Processing
  4. Symmetric Mirroring
  5. FFT Computation
  6. Magnitude Calculation
  7. Average Pooling
  8. Additional Features:
     - Zero Crossing Rate (ZCR)
     - Slope Changes
     - Dominant Frequency
  9. Feature Concatenation

## Model Architecture Options (Available on Tensorlab CLI Tools)

  1. **PIRDetection_model_1_t** (Default):
     - Compact CNN architecture
     - ~53K+ parameters
     - NPU compatible
     - Optimized for multivariate input signals

## AI Model Performance

  - **Accuracy:** ~98% (*Accuracy numbers for CC1352 with floating point feature extraction*)
  - **Accuracy:** ~92.46% (*Accuracy numbers on MSPM0 with fixed point feature extraction*)


## Training and Deployment Process

  NOTE: Running the config yaml handles everything including feature extraction, training, quantization, and compilation.

  1. Training:
     - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
     - Batch size: 64, Learning rate: 0.00001, Optimizer: Adam
     - Weight decay: 1e-20
     - Training epochs: 100

  2. Quantization:
     - INT8 quantization for NPU compatibility
     - Aggressive quantization targeting TI MCUs

  3. Compilation:
     - TI Neural Network Compiler converts trained model
     - Generates model.a and tvmgen_default.h header file
     - Hardware accelerated inference using TinyEngine NPU

## How to Run

  After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

**Windows:**
```bash
  .\run_tinyml_modelzoo.bat examples\pir_detection\config_MSPM0.yaml
 ```
**Linux:**

```bash
  ./run_tinyml_modelzoo.sh examples/pir_detection/config_MSPM0.yaml
``` 

## Available Default Configurations For Each Device Family

 
  - config.yaml - CC2755
  - config_MSPM0.yaml - MSPM0G5187
  - config_CC1352.yaml - CC1352
  - config_CC1354.yaml - CC1354
  - config_CC35X1.yaml - CC35X1

## References

  - MSPM0G5187 Technical Reference Manual https://www.ti.com/product/MSPM0G5187
  - EdgeAI Sensor Boosterpack (TIDA-010997) https://www.ti.com/tool/TIDA-010997
  - https://software-dl.ti.com/mctools/nnc/mcu/users_guide/
  - TI Model Training Guide: https://github.com/TexasInstruments/tinyml-tensorlab/tree/main
  - https://en.wikipedia.org/wiki/Passive_infrared_sensor
  - EdgeAI Software Guide: https://dev.ti.com/tirex/explore/node?node=A__AKCnvqDed-Plz2JO5Umb3Q__MSPM0-SDK__a3PaaoK__LATEST
- MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK

