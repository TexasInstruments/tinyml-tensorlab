 # Character Recognition (MNIST)

## Overview

  The Character Recognition application is an Edge AI solution that runs on the MSPM0G5187 microcontroller. It classifies single-digit handwritten character images (0-9) in real-time using the classic LeNet-5 convolutional neural network architecture.

## Problem and Solution

  - Handwritten digit recognition is a fundamental task in document processing and automation
  - Manual digit entry is error-prone and time-consuming
  - Traditional template matching approaches struggle with handwriting variability
  - Edge AI enables robust character recognition directly on resource-constrained microcontrollers

## Key Performance Targets

  - Real-time digit classification
  - High accuracy on handwritten digits
  - Low memory footprint suitable for MCU deployment
  - Simple UART-based communication with host GUI

## System Components

  1. Hardware:
     - MSPM0G5187 microcontroller [Link](https://www.ti.com/product/MSPM0G5187)
     - UART connection to host PC

  2. Software:
     - Code Composer Studio 12.x or later
     - MSPM0 SDK 2.08.00 or later

## Dataset Information

  The example uses the `mnist_image_classification` dataset based on the classic MNIST handwritten digit dataset:

  - **Source:** MNIST dataset exported via torchvision
  - **Classes:** 10 digits (0-9)
  - **Image Format:** 28x28 grayscale PNG images
  - **Download:** [mnist_classes.zip](https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/mnist_classes.zip)


For dataset recreation instructions, refer to `readme_dataset_creation.md`.

## Feature Extraction Pipeline

  1. Image Input: 28x28 grayscale image
  2. Grayscale Conversion: Single channel output
  3. Resize: Ensure 28x28 dimensions
  4. Tensor Conversion: Convert to PyTorch tensor
  5. Normalization: Mean = 0.1307, Scale = 0.3081

## Model Architecture Options (Available on Tensorlab CLI Tools)

  1. **Lenet5** (Default):
     - Classic LeNet-5 CNN architecture
     - ~60,000 parameters
     - Proven architecture for digit recognition

## Model Performance

  - Accuracy: ~99%

## Training and Deployment Process

  NOTE: Running the config yaml handles everything including feature extraction, training, quantization, and compilation.

  1. Training:
     - Use TI Edge AI Studio (GUI) or tinyml-tensorlab (CLI)
     - Batch size: 64, Learning rate: 0.1
     - Training epochs: 14

  2. Quantization:
     - INT8 quantization for reduced model size
     - Maintains accuracy while enabling MCU deployment

  3. Compilation:
     - TI Neural Network Compiler converts trained model
     - Generates model artifacts for CPU inference

## How to Run

After completing the repository setup, run the following command from the `tinyml-modelzoo` directory:

**Windows:**
```bash
  .\run_tinyml_modelzoo.bat examples\MNIST_image_classification\config_image_classification_mnist.yaml
```


**Linux:**
```bash
  ./run_tinyml_modelzoo.sh examples/MNIST_image_classification/config_image_classification_mnist.yaml
```

## References

  - MSPM0G5187 Technical Reference Manual https://www.ti.com/product/MSPM0G5187
  - https://software-dl.ti.com/mctools/nnc/mcu/users_guide/
  - TI Model Training Guide: https://github.com/TexasInstruments/tinyml-tensorlab/tree/main
  - http://yann.lecun.com/exdb/mnist/
  - https://en.wikipedia.org/wiki/LeNet
  - EdgeAI Software Guide: https://dev.ti.com/tirex/explore/node?node=A__AKCnvqDed-Plz2JO5Umb3Q__MSPM0-SDK__a3PaaoK__LATEST
- MSPM0 SDK: https://www.ti.com/tool/MSPM0-SDK
