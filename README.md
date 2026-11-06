# Tiny ML Model Optimization with PyTorch

## Overview

This is a model optimization toolkit designed for quantizing neural networks to 2-bit, 4-bit, or 8-bit precision using either Quantization Aware Training (QAT) or Post-Training Quantization (PTQ). It specifically targets Texas Instruments devices, whether they have an NPU (neural processing unit) accelerator or run on CPU. The toolkit provides wrappers for both TI-NPU optimized models (prefixed with TINPU) and CPU-compatible models (prefixed with GENERIC).

* [TinyML TorchModelOpt](./torchmodelopt/) - A collection of tools and utilities for developing embedded machine learning models in [PyTorch](https://pytorch.org), known as **model optimization tools**.
* This repository enables you to quantize your models using either Quantization Aware Training (QAT) or Post-Training Quantization (PTQ) into optimized formats that run efficiently on Texas Instruments microcontrollers.

## Installation Instructions

To use this repository as a Python package, install it directly via pip:

```commandline
pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_torchmodelopt-1.5.0-py3-none-any.whl
```

For development purposes, clone the repository and run the setup script to install the package and its dependencies:

```commandline
# Clone this repository first
git clone https://github.com/TexasInstruments/tinyml-modeloptimization.git
cd tinyml-modeloptimization/torchmodelopt
./setup.sh
```

## Verify by running an example [Motor Fault Classification]

Compact, runnable example that trains a small CNN on motor vibration data, demonstrates quantization, exports an ONNX model, and evaluates accuracy.

```commandline
cd examples/motor_fault_time_series_classification
python motor_fault_classification_tinpu_quant.py
```

## Features

The repository includes the following capabilities:

1. **Example Projects**: Ready-to-use examples to help you get started with model optimization
2. **Quantization**:
   * **Multiple Bit-Widths**: Choose lower precision (2-bit, 4-bit, or 8-bit) for weights and activations to reduce memory usage and accelerate inference
   * **Two Quantization Methods**: Support for both PTQ (Post-Training Quantization) and QAT (Quantization Aware Training)
   * **ONNX Export**: Quantized models are exported in ONNX format for easy compilation and deployment on target devices
3. **Neural Architecture Search (NAS)**: Automated tools for optimizing network architecture [Read more here](./tinyml_torchmodelopt/nas/readme.md)

Example projects are available in the [Examples](./torchmodelopt/examples/) directory.
For instructions on compiling ONNX models using the TVM Compiler, refer to the [Compilation Guide](https://software-dl.ti.com/mctools/nnc/mcu/users_guide/index.html).

## Directory Structure

```
tinyml-modeloptimization/
    ├── torchmodelopt/
        ├── examples/
            ├── audio_keyword_spotting/                     # Advanced example using Google Speech Dataset
            ├── fmnist_image_classification/                # Simple example using Fashion MNIST Dataset
            ├── mnist_lenet5_classification/                # Simple example using MNIST Dataset
            ├── motor_fault_time_series_classification/     # Intermediate example using Motor Fault Dataset
            ├── torque_time_series_regression/              # Simple example demonstrating regression workflow
            ├── README.md    
        ├── requirements/
            ├── requirements.txt
        ├── tinyml_torchmodelopt/
            ├── nas/                                        # Neural Architecture Search tools
            ├── quantization/                               # Quantization implementation (PTQ/QAT)
                ├── base/                                   # Base wrapper class for quantized models
                ├── generic/                                # CPU quantization constraints and utilities
                    ├── quant_fx.py
                    ├── quant_utils.py
                ├── tinpu/                                  # TI-NPU quantization constraints and utilities
                    ├── quant_fx.py
                    ├── quant_utils.py
            ├── surgery/                                    # Utilities for replacing and modifying PyTorch layers
        ├── setup.py
        ├── setup.sh
    ├── LICENSE
    ├── README.md
```

**Note**: Some files and directories have been omitted from this structure to keep it concise and focused on the most important components.



# Using Quantization Wrappers for your PyTorch model

This Python package provides quantization wrappers that allow you to pass your PyTorch model with specific configurations and receive a quantized model. The quantization wrappers are organized based on two factors: the Quantization Method (QAT/PTQ) and the target device on which the model will run (GENERIC/TINPU).

An important class, **TinyMLQConfigType**, helps you create a Quantization Configuration (or qconfig) to direct the wrapper on how to perform quantization.

Based on the quantization method and target device, the quantization wrappers are available in four variants:
- GenericTinyMLQATFxModule
- GenericTinyMLPTQFxModule
- TINPUTinyMLQATFxModule
- TINPUTinyMLPTQFxModule


### 1. TI optimized acceleration - Devices with Hardware based TI NPU or even without it! (TINPUTinyMLQATFxModule / TINPUTinyMLPTQFxModule )

- TINPUTinyMLQATFxModule / TINPUTinyMLPTQFxModule is a PyTorch Quantization Wrapper module that incorporates the constraints of the TI NPU Hardware accelerator. We call this a wrapper as it wraps your PyTorch module and induces the constraints of the hardware.
- It can be imported as follows `from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule`

The following is a sample usage of how to incorporate this module. 

```python
from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule, TINPUTinyMLPTQFxModule

# create your model here:
model = ...

# load your pretrained checkpoint/weights here or run your usual floating-point training
pretrained_data = torch.load(pretrained_path)
model.load_state_dict(pretrained_data)

# create a dummy input - this is required for onnx export - will vary depending on your model.
example_inputs = torch.rand((1,1,256,1))

qconfig_type = TinyMLQConfigType(weight_bitwidth=8, activation_bitwidth=8, auto_quantization=False)
# wrap your model in TINPUTinyMLQATFxModule / TINPUTinyMLPTQFxModule
model = TINPUTinyMLQATFxModule(model, example_inputs=example_inputs, qconfig_type=qconfig_type, total_epochs=epochs)
# model = TINPUTinyMLPTQFxModule(model, example_inputs=example_inputs, qconfig_type=qconfig_type, total_epochs=epochs)

# train the wrapped model in your training loop here with loss, backward, optimizer, etc.
# your usual training loop
model.train()
for e in range(epochs):
    for images, target in my_dataset_train:
        output = model(images)
        # loss, backward(), optimizer step, etc comes here as usual in training

model.eval()

# convert the model to operate with integer operations (instead of QDQ FakeQuantize operations)
model = model.convert()

# export the quantized model to onnx format
model.export(dummy_input, os.path.join(save_path,'model_int8.onnx'), input_names=['input'])
```


### 2. Wrapper for standard Generic Pytorch style quantization (GenericTinyMLQATFxModule / GenericTinyMLPTQFxModule)

- GenericTinyMLQATFxModule / GenericTinyMLPTQFxModule is a PyTorch module that incorporates the constraints of typical INT8 quantization in PyTorch. This makes use of the PyTorch quantization APIs, but makes it easy to do QAT with minimal code changes. For more details of [PyTorch quantization, see its documentation](https://pytorch.org/docs/stable/quantization.html)
- The wrapper module can be imported as follows `from tinyml_torchmodelopt.quantization import GenericTinyMLQATFxModule`

The following is a sample usage of how to incorporate this module. 

```python
from tinyml_torchmodelopt.quantization import GenericTinyMLQATFxModule, GenericTinyMLPTQFxModule

# create your model here:
model = ...

# load your pretrained checkpoint/weights here or run your usual floating-point training
pretrained_data = torch.load(pretrained_path)
model.load_state_dict(pretrained_data)

# create a dummy input - this is required for onnx export - will change depending on your model.
example_inputs = torch.rand((1,1,256,1))

qconfig_type = TinyMLQConfigType(weight_bitwidth=8, activation_bitwidth=8, auto_quantization=False)
# wrap your model in GenericTinyMLQATFxModule / GenericTinyMLPTQFxModule
model = GenericTinyMLQATFxModule(model, example_inputs=example_inputs, qconfig_type=qconfig_type, total_epochs=epochs)
# model = GenericTinyMLPTQFxModule(model, example_inputs=example_inputs, qconfig_type=qconfig_type, total_epochs=epochs)

# train the wrapped model in your training loop here with loss, backward, optimizer, etc.
# your usual training loop
model.train()
for e in range(epochs):
    for images, target in my_dataset_train:
        output = model(images)
        # loss, backward(), optimizer step, etc comes here as usual in training

model.eval()

# convert the model to operate with integer operations (instead of QDQ FakeQuantize operations)
model = model.convert()

# export the quantized model to onnx format
model.export(dummy_input, os.path.join(save_path,'model_int8.onnx'), input_names=['input'])
```

### Evaluate your Model before running on device

You can now use `model` for evaluation before compiling and running on device

```python
import onnxruntime as ort

model_name = 'model_int8.onnx'
example_inputs = torch.rand((1,1,256,1))

ort_session_options = ort.SessionOptions()
ort_session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

ort_session = ort.InferenceSession(model_name, ort_session_options)
prediction = ort_session.run(None, {INPUT_NAME: example_inputs})

print(prediction)
```

## Wrappers Provided

**TinyMLQuantFxBaseModule** is the base wrapper class for all four quantization wrapper variations.

### 1. GENERIC Wrappers for CPU Quantization
* **GenericTinyMLQATFxModule** & **GenericTinyMLPTQFxModule**
    - Runs on CPU
    - Model is present in ONNX & QDQ Format
    - Quantized according to QAT/PTQ

### 2. TINPU Wrappers for NPU Quantization
* **TINPUTinyMLQATFxModule** & **TINPUTinyMLPTQFxModule**
    - Runs on NPU (can also run on CPU)
    - Model is present in ONNX TINPU Format
    - Quantized according to QAT/PTQ


## Options present in Quantization Wrapper

```python

ti_model = TinyMLQuantFxBaseModule(model, 
                                   total_epochs=10, 
                                   qconfig_type=TinyMLQConfigType(weight_bitwidth=8, activation_bitwidth=8, auto_quantization=False),
                                   example_inputs=torch.rand((1,1,256,1)),
                                   is_qat=True,
                                   model_output_format=TinyMLModelQConfigFormat.TINPU_INT_MODEL
                                   num_batch_norm_update_epochs=None, 
                                   num_observer_update_epochs=None,
                                   bias_calibration_factor=0.0,
                                   output_int=False
                                   verbose=False)

```

#### Argument Descriptions

|             Argument                  |            Type           | Description |
|---------------------------------------|---------------------------|-------------|
| **model**                             | torch.nn.Module           | Model |
| **total_epochs**                      | int                       | Total number of quantized training epochs |
| **qconfig_type**                      | TinyMLQConfigType         | QConfig configurations for model quantization |
| **example_inputs**                    | torch.Tensor              | Example input with batch size 1|
| **is_qat**                            | bool                      | Toggle for PTQ / QAT |
| **model_output_format**               | TinyMLModelQConfigFormat  | Output format of the onnx model, INT/TINPU_INT_MODEL |
| **num_batch_norm_update_epochs**      | bool/int/None             | Whether freezing BatchNorm allowed or not, if yes, then provide number of epochs after freezing happens |
| **num_observer_update_epochs**        | bool/int/None             | Whether freezing observers allowed or not, if yes, then provide number of epochs after freezing happens |
| **bias_calibration_factor**           | float                     | Calibrate bias with a scaling factor  |
| **output_int**                        | bool                      | Obtain the quantized or dequantized output |
| **verbose**                           | bool                      | Enable or disable verbose statements |


## Examples for Training and Quantization
Detailed examples for using this repository is present at [examples](./torchmodelopt/examples/). From simple to advanced examples are provided. You can select the example based on what you are looking for.
