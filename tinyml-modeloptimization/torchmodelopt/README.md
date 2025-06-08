## Installation Instructions

If you want to use the repository as it is, i.e a Python package, then you can simply install this as a pip installable package:

```commandline
pip install git+https://github.com/TexasInstruments/tinyml-tensorlab.git@main#subdirectory=tinyml-modeloptimization/torchmodelopt
```

To setup the repository for development, this python package and the dependencies can be installed by using the setup file.

```commandline
./setup.sh
```

## Quantization Aware Training (QAT)
This repository helps you to quantize (with Quantization Aware Training - QAT) your model to formats that can run optimally on TI's MCUs

### QAT for C2000 / ARM devices with Hardware based TI NPU acceleration
Note: Please consult the device and SDK documentation to understand whether Hardware based TI NPU acceleration is supported in that device. 

#### TINPUTinyMLQATFxModule 
TINPUTinyMLQATFxModule is a PyTorch module that incorporates the constraints of TI NPU Hardware accelerator. We call this a QAT wrapper as it wraps your PyTorch module and induces the constraints of the hardware.

It can be imported as follows.
from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule

The following is a sample usage of how to incorporate this module. 

```python
from tinyml_torchmodelopt.quantization import TINPUTinyMLQATFxModule

# create your model here:
model = ...

# load your pretrained checkpoint/weights here or run your usual floating-point training
pretrained_data = torch.load(pretrained_path)
model.load_state_dict(pretrained_data)

# wrap your model in TINPUTinyMLQATFxModule
model = TINPUTinyMLQATFxModule(model, total_epochs=epochs)

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

# create a dummy input - this is required for onnx export - will change depending on your model.
dummy_input = torch.rand((1,1,256,1))

# export the quantized model to onnx format
torch.onnx.export(model.module, dummy_input, os.path.join(save_path,'model_int8.onnx'), input_names=['input'])
```


### QAT for C2000 / ARM MCUs without using Hardware based TI NPU acceleration

#### GenericTinyMLQATFxModule 
GenericTinyMLQATFxModule is a PyTorch module that incorporates the constraints of typical INT8 quantization in PyTorch. This makes use of the PyTorch quantization APIs, but makes it easy to do QAT with minimal code changes. For more details of [PyTorch quantization, see its documentation](https://pytorch.org/docs/stable/quantization.html)

The wrapper module can be imported as follows.
from tinyml_torchmodelopt.quantization import GenericTinyMLQATFxModule

The following is a sample usage of how to incorporate this module. 

```python
from tinyml_torchmodelopt.quantization import GenericTinyMLQATFxModule

# create your model here:
model = ...

# load your pretrained checkpoint/weights here or run your usual floating-point training
pretrained_data = torch.load(pretrained_path)
model.load_state_dict(pretrained_data)

# wrap your model in TINPUTinyMLQATFxModule
model = GenericTinyMLQATFxModule(model, total_epochs=epochs)

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

# create a dummy input - this is required for onnx export - will change depending on your model.
dummy_input = torch.rand((1,1,256,1))

# export the quantized model to onnx format
torch.onnx.export(model.module, dummy_input, os.path.join(save_path,'model_int8.onnx'), input_names=['input'])
```


## Examples for Training and QAT
Detailed examples for using this repository is present at [examples](./examples/)
