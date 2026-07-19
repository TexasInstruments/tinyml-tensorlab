# Dynamic Hand Gesture Recognition

## Overview

  The Dynamic Hand Gesture Recognition application is an Edge AI solution that runs on the MSPM0G5187 microcontroller with integrated Neural Processing Unit (NPU). It classifies dynamic hand gestures in real-time using 3-axis accelerometer data from TI's Sensor BoosterPack, enabling intuitive gesture-based human-machine interfaces on resource-constrained embedded devices.

## Problem and Solution

  - Traditional human-machine interfaces rely on physical buttons or touchscreens, which are impractical in hands-free or hygienic environments
  - Manual gesture recognition systems require complex rule-based logic that is brittle to natural variation in user movements
  - Cloud-based gesture recognition introduces latency and privacy concerns for real-time control applications
  - Edge AI enables robust, low-latency gesture classification directly on the microcontroller without any cloud connectivity

## Key Performance Targets

  - Real-time gesture classification from accelerometer data
  - 94.46% classification accuracy on test data
  - Low memory footprint suitable for MCU deployment
  - 4-class gesture discrimination using 3-axis accelerometer input

## System Components

  1. Hardware:
     - MSPM0G5187 microcontroller with integrated NPU [Link](https://www.ti.com/product/MSPM0G5187)
     - TI Sensor BoosterPack (provides 3-axis accelerometer input)

  2. Software:
     - Code Composer Studio 12.x or later
     - MSPM0 SDK 2.10.04 or later
     - TI Edge AI Studio

## Dataset Information

  The example uses the `Hand_gesture_dataset` dataset containing labeled 3-axis accelerometer recordings of dynamic hand gestures:

  - **Classes:**
    - Class 1 — Circle (both clockwise and counter-clockwise)
    - Class 2 — Wave
    - Class 3 — Tap
    - Class 4 — Others (non-gesture or unrecognized motion)
  - **Sensor:** 3-axis accelerometer (X, Y, Z) from TI Sensor BoosterPack
  - **Download:** [hand_gesture_dataset.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/01_04_00/datasets/hand_gesture_dataset.zip)

## Feature Extraction Pipeline

  1. Accelerometer Input: 3-axis (X, Y, Z) raw sensor data from Sensor BoosterPack
  2. Windowing: 256 samples per frame
  3. Stride: 0.25 (25% of frame size — frames overlap with a step of 64 samples)
  4. Normalization: Range normalization applied per frame
  5. Single frame input to model

## Model Architecture Options (Available on Tensorlab CLI Tools)

  1. **CLS_55k_NPU** (Default):
     - CNN architecture optimized for NPU execution
     - ~55K parameters
     - NPU compatible for low-latency inference

## Model Performance

  - Accuracy: ~94.46% on test data

## Training and Deployment Process

  NOTE: Running the config yaml handles everything including feature extraction, training, quantization, and compilation.

  1. Training:
     - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
     - Batch size: 30, Learning rate: 0.001, Optimizer: Adam
     - Training epochs: 40

  2. Quantization:
     - INT8 quantization enabled by default
     - Maintains accuracy while reducing model size for MCU deployment

  3. Compilation:
     - TI Neural Network Compiler converts trained model
     - Generates model artifacts optimized for NPU execution

## How to Run

  After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

  **Windows:**
  ```bash
  .\run_tinyml_modelzoo.bat examples\dynamic_hand_gesture_recognition\config_MSPM0.yaml
  ```

  **Linux:**
  ```bash
  ./run_tinyml_modelzoo.sh examples/dynamic_hand_gesture_recognition/config_MSPM0.yaml
  ```

## References

  - MSPM0G5187 Technical Reference Manual https://www.ti.com/product/MSPM0G5187
  - TI Sensor BoosterPack https://www.ti.com/tool/BOOSTXL-SENSORS
  - https://software-dl.ti.com/mctools/nnc/mcu/users_guide/
  - TI Model Training Guide: https://github.com/TexasInstruments/tinyml-tensorlab/tree/main
  - EdgeAI Software Guide: https://dev.ti.com/tirex/explore/node?node=A__AKCnvqDed-Plz2JO5Umb3Q__MSPM0-SDK__a3PaaoK__LATEST
  - MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
