## Using Quantization Wrappers for your PyTorch model

This repository helps you to quantize your model to formats that can run optimally on TI's MCUs

**Note**: Please consult the device and SDK documentation to understand whether Hardware based TI NPU acceleration is supported in that device. 

### 1. TI optimised acceleration - C2000 / ARM core devices with Hardware based TI NPU or even without it! (TINPUTinyMLQATFxModule / TINPUTinyMLPTQFxModule )

- TINPUTinyMLQATFxModule / TINPUTinyMLPTQFxModule is a PyTorch module that incorporates the constraints of TI NPU Hardware accelerator. We call this a wrapper as it wraps your PyTorch module and induces the constraints of the hardware.
- It can be imported as follows.
from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule

The following is a sample usage of how to incorporate this module. 

```python
from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule, TINPUTinyMLPTQFxModule

# create your model here:
model = ...

# load your pretrained checkpoint/weights here or run your usual floating-point training
pretrained_data = torch.load(pretrained_path)
model.load_state_dict(pretrained_data)

# create a dummy input - this is required for onnx export - will change depending on your model.
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
- The wrapper module can be imported as follows.
from tinyml_torchmodelopt.quantization import GenericTinyMLQATFxModule

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

### 1. Base Wrapper for Native PyTorch Quantization
* **TinyMLQuantFxBaseModule**
    - Base class for Generic and TINPU wrappers
    - Model is present in ONNX Format
    - Quantized according to **is_qat** option

### 2. GENERIC Wrappers for CPU Quantization
* **GenericTinyMLQATFxModule** & **GenericTinyMLPTQFxModule**
    - Runs on CPU
    - Model is present in ONNX QDQ Format
    - Quantized according to QAT/PTQ

### 3. TINPU Wrappers for NPU Quantization
* **TINPUTinyMLQATFxModule** & **TINPUTinyMLPTQFxModule**
    - Runs on NPU
    - Model is present in ONNX TINPU Format
    - Quantized according to QAT/PTQ


## Options present in TinyMLQuantFxBaseModule

```python

ti_model = TinyMLQuantFxBaseModule(model, 
                                   total_epochs=10, 
                                   qconfig_type=TinyMLQConfigType(weight_bitwidth=8, activation_bitwidth=8, auto_quantization=False)
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
| **bias_calibration_factor**           | float                     | Use bias calibration |
| **output_int**                        | bool                      | Obtain the quantized or dequantized output |
| **verbose**                           | bool                      | Enable or disable verbose statements |


## Examples for Training and Quantization
Detailed examples for using this repository is present at [examples](./examples/). From simple to advanced examples are provided. You can select the example based on what you are looking for.
