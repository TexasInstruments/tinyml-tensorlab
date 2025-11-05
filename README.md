# Tiny ML Model Optimization with PyTorch

## Overview
Model optimization toolkit that is necessary for quantization for 2bit/4bit/8bit weights in QAT(Quantization Aware Training)/PTQ(Post Training Quantization) flows for TI devices with or without NPU. The following package provides wrappers for models running on TI-NPU (HW Accelerator) or CPU. The wrappers for TI-NPU starts from TINPU whereas for CPU starts from GENERIC.

* [Tiny ML TorchModelOpt Documentation](./torchmodelopt/) - Tools and utilities to help the development of embedded Models in [Pytorch](https://pytorch.org) - we call these **model optimization tools**.
* This repository helps you to quantize (with Quantization Aware Training - QAT or Post Training Quantization - PTQ) your model to formats that can run optimally on TI's MCUs

## Installation Instructions

If you want to use the repository as it is, i.e a Python package, then you can simply install this as a pip installable package:

```commandline
pip install git+https://github.com/TexasInstruments/tinyml-tensorlab.git@r1.2#subdirectory=tinyml-modeloptimization/torchmodelopt
```

To setup the repository for development, this python package and the dependencies can be installed by using the setup file.

```commandline
cd tinyml-modeloptimization/torchmodelopt
./setup.sh
```

## Features

The repository provides the following features:
1. **Examples**: The repository comes with examples to get started with modeloptimization
2. **Quantization** :
   * **Different Bit-Widths**: Lower precision for representing weights, biases and numbers can be selected to save memory and speed up the inference
   * **PTQ/QAT**: Different quantization methods like PTQ and QAT are supported
   * **ONNX Models**: Quantized models are exported as ONNX structure which can be easily compiled and run on device
*  **Neural network Architecture Search (NAS)**: [Read more here](./tinyml_torchmodelopt/nas/readme.md)

Examples for using this repository is present at [Examples](./torchmodelopt/examples/) and for compilation of ONNX Models using TVM Compiler at [Compilation](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/index.html)

## Directory Structure

```
tinyml-modeloptimization/
    ├── torchmodelopt/
        ├── examples/
            ├── audio_keyword_spotting/                     # Advanced example on Google Speech Dataset
            ├── fmnist_image_classification/                # Simple example on Fashion MNIST Dataset
            ├── motor_fault_time_series_classification/     # Moderate example on MotorFault Dataset
            ├── README.md    
        ├── requirements/
            ├── requirements.txt
        ├── tinyml_torchmodelopt/
            ├── nas/                                        # Neural Architecture Search
            ├── quantization/                               # Scripts handling PTQ/QAT quantization
                ├── base/                                   # Base Wrapper for Quantized Model
                ├── generic/                                # Constraints for CPU Quantized Model
                    ├── quant_fx.py
                    ├── quant_utils.py
                ├── tinpu/                                  # Constraints for NPU Quantized Model
                    ├── quant_fx.py
                    ├── quant_utils.py
            ├── surgery/                                    # Scripts for replacing PyTorch layers
        ├── setup.py
        ├── setup.sh
    ├── LICENSE
    ├── README.md
```

**Note**: Some files and folders are not represented in this dir structure to avoid cluttering and removing unnecessary information.

---